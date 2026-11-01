"""Reconstruct an object from the observation-only Isaac orbit dataset.

The program deliberately refuses to read ``evaluator_only``.  Every method
uses RGB, noisy depth, reported camera poses, and camera intrinsics only.

Implemented experimental candidates:

* tsdf_fk: deterministic RGB-D TSDF using the perturbed FANUC/FK pose prior.
* tsdf_icp: bounded per-view point-to-plane refinement followed by TSDF.
* poisson_depth: fused oriented RGB-D points and screened Poisson.
* laser_stripe_poisson: idealized 2-D line-profiler analogue (narrow depth
  stripe only), clearly not a substitute for testing physical laser hardware.
* visual_hull: RGB silhouette space carving; depth is used only to establish
  a conservative reconstruction volume.
* nbv_tsdf: observation-driven greedy next-view selection followed by TSDF.
* dual_pass_tsdf: TSDF using both upright and flipped scan passes.

All output meshes are emitted as PLY, OBJ, and STL when the format supports
the reconstructed topology.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation
from skimage import measure


METHODS = (
    "tsdf_fk",
    "tsdf_icp",
    "poisson_depth",
    "laser_stripe_poisson",
    "visual_hull",
    "nbv_tsdf",
    "dual_pass_tsdf",
    "photometric_poisson",
    "touch_scaled_tsdf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--voxel-mm", type=float, default=0.75)
    parser.add_argument("--poisson-depth", type=int, default=9)
    parser.add_argument("--nbv-views", type=int, default=24)
    parser.add_argument("--visual-hull-mm", type=float, default=1.25)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def load_manifest(root: Path) -> dict:
    path = root / "observations" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    forbidden = root / "evaluator_only"
    if forbidden.resolve() in path.resolve().parents:
        raise RuntimeError("reconstruction attempted to access evaluator-only data")
    return manifest


def intrinsics(manifest: dict) -> tuple[o3d.camera.PinholeCameraIntrinsic, dict]:
    cam = manifest["camera"]
    if "resolution" in cam:
        width, height = map(int, cam["resolution"])
    else:
        width, height = int(cam["width"]), int(cam["height"])
    focal = float(cam["focal_length_mm"])
    hap = float(cam["horizontal_aperture_mm"])
    vap = float(cam["vertical_aperture_mm"])
    fx, fy = focal / hap * width, focal / vap * height
    cx, cy = width * 0.5, height * 0.5
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy), {
        "width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "tan_h": hap / (2.0 * focal), "tan_v": vap / (2.0 * focal),
    }


def usd_row_to_opencv_object(matrix: np.ndarray) -> np.ndarray:
    """Return column-vector T_object_from_camera(OpenCV).

    USD/Isaac camera coordinates are +X right, +Y up, -Z forward and the
    supplied matrix uses row vectors. Open3D uses +X right, +Y down, +Z
    forward and column vectors.
    """
    usd_row = np.asarray(matrix, dtype=np.float64)
    cv_to_usd = np.eye(4)
    cv_to_usd[:3, :3] = np.diag((1.0, -1.0, -1.0))
    cv_row_to_object = cv_to_usd @ usd_row
    return cv_row_to_object.T


def read_rgb_depth(obs: Path, frame: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(obs / frame["rgb"]), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(obs / frame["rgb"])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = np.load(obs / frame["depth"])["depth"].astype(np.float32)
    # The synthetic part has a calibrated blue finish. This is background
    # subtraction, not a semantic target label or an evaluator-only mask.
    f = rgb.astype(np.float32) + 1.0
    chroma = f / f.sum(axis=2, keepdims=True)
    mask = (chroma[..., 2] > 0.44) & (f[..., 2] > 1.12 * f[..., 1]) & (f[..., 2] > 1.25 * f[..., 0])
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    clean_depth = depth.copy()
    clean_depth[~mask | ~np.isfinite(clean_depth) | (clean_depth <= 0)] = 0.0
    return rgb, clean_depth, mask


def make_rgbd(rgb: np.ndarray, depth: np.ndarray) -> o3d.geometry.RGBDImage:
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(rgb.astype(np.uint8))),
        o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.float32))),
        depth_scale=1.0,
        depth_trunc=1.5,
        convert_rgb_to_intensity=False,
    )


def select_frames(frames: list[dict], max_frames: int, include_flipped: bool) -> list[dict]:
    selected = frames if include_flipped else [f for f in frames if f["pass"] == "upright"]
    if max_frames and len(selected) > max_frames:
        indices = np.linspace(0, len(selected) - 1, max_frames).round().astype(int)
        selected = [selected[i] for i in indices]
    return selected


def camera_pcd(obs: Path, frame: dict, intrinsic: o3d.camera.PinholeCameraIntrinsic,
               stripe_half_width: int | None = None) -> tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray, np.ndarray]:
    rgb, depth, mask = read_rgb_depth(obs, frame)
    if stripe_half_width is not None:
        keep = np.zeros_like(mask)
        center = depth.shape[1] // 2
        keep[:, max(0, center - stripe_half_width): center + stripe_half_width + 1] = True
        depth[~keep] = 0.0
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(make_rgbd(rgb, depth), intrinsic)
    transform = usd_row_to_opencv_object(np.asarray(frame["camera_to_object"]))
    pcd.transform(transform)
    return pcd, rgb, depth, mask


def clean_o3d_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def export_mesh(mesh: o3d.geometry.TriangleMesh | trimesh.Trimesh, out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        mesh = clean_o3d_mesh(mesh)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        colours = np.asarray(mesh.vertex_colors)
        tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if len(colours) == len(vertices):
            tri.visual.vertex_colors = np.clip(colours * 255.0, 0, 255).astype(np.uint8)
    else:
        tri = mesh.copy()
        tri.remove_unreferenced_vertices()
    paths = {}
    for suffix in ("ply", "obj", "stl"):
        path = out.with_suffix(f".{suffix}")
        tri.export(path)
        paths[suffix] = path.name
    return {
        "outputs": paths,
        "vertices": int(len(tri.vertices)),
        "faces": int(len(tri.faces)),
        "watertight": bool(tri.is_watertight),
        "components": int(len(tri.split(only_watertight=False))),
    }


def bounded_icp_transforms(obs: Path, frames: list[dict], intrinsic, voxel: float) -> tuple[list[np.ndarray], dict]:
    clouds = []
    priors = []
    for frame in frames:
        pcd, _, _, _ = camera_pcd(obs, frame, intrinsic)
        pcd = pcd.voxel_down_sample(max(voxel * 1.5, 0.0010))
        if len(pcd.points):
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.006, max_nn=40))
        clouds.append(pcd)
        priors.append(usd_row_to_opencv_object(np.asarray(frame["camera_to_object"])))

    reference = o3d.geometry.PointCloud()
    for cloud in clouds[::max(1, len(clouds) // 16)]:
        reference += cloud
    reference = reference.voxel_down_sample(max(voxel, 0.0007))
    reference.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=50))

    corrected = []
    accepted = 0
    rejected = 0
    translations = []
    rotations = []
    for cloud, prior in zip(clouds, priors):
        if len(cloud.points) < 100:
            corrected.append(prior)
            rejected += 1
            continue
        result = o3d.pipelines.registration.registration_icp(
            cloud, reference, 0.0035, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=35),
        )
        correction = result.transformation
        tnorm = float(np.linalg.norm(correction[:3, 3]))
        angle = float(np.degrees(Rotation.from_matrix(correction[:3, :3]).magnitude()))
        if result.fitness >= 0.40 and tnorm <= 0.0015 and angle <= 0.45:
            corrected.append(correction @ prior)
            accepted += 1
            translations.append(tnorm * 1000.0)
            rotations.append(angle)
        else:
            corrected.append(prior)
            rejected += 1
    return corrected, {
        "accepted": accepted,
        "rejected": rejected,
        "translation_correction_mm_mean": float(np.mean(translations)) if translations else 0.0,
        "rotation_correction_deg_mean": float(np.mean(rotations)) if rotations else 0.0,
    }


def integrate_tsdf(obs: Path, frames: list[dict], intrinsic, voxel: float,
                   transforms: list[np.ndarray] | None = None) -> o3d.geometry.TriangleMesh:
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=4.0 * voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for index, frame in enumerate(frames):
        rgb, depth, _ = read_rgb_depth(obs, frame)
        object_from_cam = transforms[index] if transforms is not None else usd_row_to_opencv_object(np.asarray(frame["camera_to_object"]))
        volume.integrate(make_rgbd(rgb, depth), intrinsic, np.linalg.inv(object_from_cam))
    return clean_o3d_mesh(volume.extract_triangle_mesh())


def fused_points(obs: Path, frames: list[dict], intrinsic, voxel: float,
                 stripe_half_width: int | None = None) -> o3d.geometry.PointCloud:
    merged = o3d.geometry.PointCloud()
    for frame in frames:
        pcd, _, _, _ = camera_pcd(obs, frame, intrinsic, stripe_half_width)
        if len(pcd.points) == 0:
            continue
        pcd = pcd.voxel_down_sample(max(voxel, 0.0005))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=40))
        eye = np.asarray(frame["eye_object_m"], dtype=np.float64)
        pcd.orient_normals_towards_camera_location(eye)
        merged += pcd
    merged = merged.voxel_down_sample(voxel)
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.2)
    return merged


def photometric_points(obs: Path, frames: list[dict], intrinsic, calib: dict, voxel: float) -> tuple[o3d.geometry.PointCloud, dict]:
    """Fuse noisy depth points with four-light photometric normals.

    The four calibrated lights are fixed in the world/object frame for the
    upright pass. A per-frame distant-light approximation is used; it is
    intentionally simple and its performance is measured rather than assumed.
    """
    light_positions = np.asarray(((0.35, 0.0, 0.35), (-0.35, 0.0, 0.35),
                                  (0.0, 0.35, 0.32), (0.0, -0.35, 0.32)), dtype=np.float64)
    merged = o3d.geometry.PointCloud()
    used = 0
    for frame in frames:
        photos = frame.get("photometric_rgb")
        if frame["pass"] != "upright" or not photos or len(photos) < 4:
            continue
        rgb, depth, _ = read_rgb_depth(obs, frame)
        ys, xs = np.nonzero(depth > 0)
        if len(xs) < 100:
            continue
        z = depth[ys, xs].astype(np.float64)
        camera_points = np.column_stack(((xs - calib["cx"]) / calib["fx"] * z,
                                         (ys - calib["cy"]) / calib["fy"] * z, z))
        transform = usd_row_to_opencv_object(np.asarray(frame["camera_to_object"]))
        object_points = camera_points @ transform[:3, :3].T + transform[:3, 3]
        centre = np.median(object_points, axis=0)
        lights = light_positions - centre
        lights /= np.linalg.norm(lights, axis=1, keepdims=True)
        intensities = []
        for name in photos[:4]:
            bgr = cv2.imread(str(obs / name), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(obs / name)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
            intensities.append(gray[ys, xs])
        values = np.column_stack(intensities)
        values -= np.min(values, axis=1, keepdims=True)
        g = values @ np.linalg.pinv(lights).T
        norm = np.linalg.norm(g, axis=1)
        valid = norm > 1e-5
        if valid.sum() < 100:
            continue
        object_points = object_points[valid]
        normals = g[valid] / norm[valid, None]
        eye = np.asarray(frame["eye_object_m"])
        toward_camera = eye - object_points
        flip = np.einsum("ij,ij->i", normals, toward_camera) < 0
        normals[flip] *= -1.0
        colours = rgb[ys[valid], xs[valid]].astype(np.float64) / 255.0
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(object_points)
        cloud.normals = o3d.utility.Vector3dVector(normals)
        cloud.colors = o3d.utility.Vector3dVector(colours)
        merged += cloud.voxel_down_sample(max(voxel, 0.0005))
        used += 1
    merged = merged.voxel_down_sample(voxel)
    if len(merged.points):
        merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    return merged, {"photometric_keyframes_used": used, "photometric_points": int(len(merged.points))}


def poisson_mesh(points: o3d.geometry.PointCloud, depth: int) -> tuple[o3d.geometry.TriangleMesh, dict]:
    if len(points.points) < 500:
        raise RuntimeError(f"too few points for Poisson: {len(points.points)}")
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        points, depth=depth, width=0, scale=1.08, linear_fit=True,
    )
    density = np.asarray(density)
    if len(density):
        threshold = float(np.quantile(density, 0.015))
        mesh.remove_vertices_by_mask(density < threshold)
    bbox = points.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.02, bbox.get_center())
    mesh = mesh.crop(bbox)
    return clean_o3d_mesh(mesh), {"input_points": int(len(points.points))}


def rgb_mask(obs: Path, frame: dict) -> np.ndarray:
    return read_rgb_depth(obs, frame)[2]


def project_object_points(points: np.ndarray, camera_to_object_row: np.ndarray, calib: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(camera_to_object_row, dtype=np.float64)
    eye = matrix[3, :3]
    rotation_rows = matrix[:3, :3]
    camera = (points - eye) @ rotation_rows.T
    forward = -camera[:, 2]
    u = (camera[:, 0] / np.maximum(forward, 1e-9) / calib["tan_h"] + 1.0) * 0.5 * calib["width"]
    v = (1.0 - camera[:, 1] / np.maximum(forward, 1e-9) / calib["tan_v"]) * 0.5 * calib["height"]
    return u, v, forward


def conservative_bounds(obs: Path, frames: list[dict], intrinsic, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    cloud = fused_points(obs, frames[::max(1, len(frames) // 24)], intrinsic, max(voxel * 2, 0.002))
    xyz = np.asarray(cloud.points)
    if len(xyz) < 100:
        raise RuntimeError("insufficient depth points to establish a reconstruction volume")
    lo = np.quantile(xyz, 0.003, axis=0) - np.asarray((0.012, 0.012, 0.012))
    hi = np.quantile(xyz, 0.997, axis=0) + np.asarray((0.012, 0.012, 0.012))
    lo[2] = min(lo[2], -0.003)
    return lo, hi


def visual_hull_mesh(obs: Path, frames: list[dict], intrinsic, calib: dict, spacing: float) -> tuple[trimesh.Trimesh, dict]:
    # A depth-derived conservative box is legal sensor input; the per-voxel
    # carving itself consumes RGB silhouettes only.
    lo, hi = conservative_bounds(obs, frames, intrinsic, spacing)
    dims = np.maximum(3, np.ceil((hi - lo) / spacing).astype(int) + 1)
    # Bound memory if an outlier depth expands the volume unexpectedly.
    if int(np.prod(dims)) > 8_000_000:
        scale = (np.prod(dims) / 8_000_000.0) ** (1.0 / 3.0)
        spacing *= scale
        dims = np.maximum(3, np.ceil((hi - lo) / spacing).astype(int) + 1)
    gx, gy, gz = np.meshgrid(
        np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing="ij"
    )
    points = lo + np.column_stack((gx.ravel(), gy.ravel(), gz.ravel())) * spacing
    occupied = np.ones(len(points), dtype=bool)
    # 24-36 well-spaced silhouettes are ample for the visual hull and make
    # the method CPU-tractable.
    take = min(36, len(frames))
    indices = np.linspace(0, len(frames) - 1, take).round().astype(int)
    for idx in indices:
        frame = frames[int(idx)]
        mask = rgb_mask(obs, frame)
        u, v, forward = project_object_points(points, np.asarray(frame["camera_to_object"]), calib)
        ui, vi = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
        valid = (forward > 0.01) & (ui >= 0) & (ui < calib["width"]) & (vi >= 0) & (vi < calib["height"])
        inside = np.zeros(len(points), dtype=bool)
        inside[valid] = mask[vi[valid], ui[valid]]
        occupied &= inside
        if not occupied.any():
            raise RuntimeError(f"visual hull emptied at frame {frame['id']}")
    volume = occupied.reshape(tuple(dims))
    verts, faces, normals, _ = measure.marching_cubes(volume.astype(np.float32), 0.5, spacing=(spacing,) * 3)
    verts += lo
    tri = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    return tri, {
        "grid_spacing_mm": spacing * 1000.0,
        "grid_dimensions": dims.tolist(),
        "occupied_voxels": int(occupied.sum()),
        "silhouette_views": int(take),
        "depth_use": "conservative volume only",
    }


def angular_feature(frame: dict) -> np.ndarray:
    eye = np.asarray(frame["eye_object_m"], dtype=np.float64)
    return eye / max(np.linalg.norm(eye), 1e-9)


def nbv_select(frames: list[dict], count: int) -> list[dict]:
    """Greedy uncertainty/novelty proxy using only reported sensor poses.

    Four cardinal mid-elevation seed views establish the model. Remaining
    poses maximize their minimum angular baseline with a mild high-elevation
    bonus, a deterministic proxy for expected unseen-surface information.
    """
    upright = [f for f in frames if f["pass"] == "upright"]
    count = min(max(4, count), len(upright))
    target_el = min({float(f["elevation_deg"]) for f in upright}, key=lambda x: abs(x - 32.0))
    mid = [f for f in upright if float(f["elevation_deg"]) == target_el]
    seeds = []
    for target_az in (0, 90, 180, 270):
        seeds.append(min(mid, key=lambda f: abs(((float(f["azimuth_deg"]) - target_az + 180) % 360) - 180)))
    selected = list(dict.fromkeys(f["id"] for f in seeds))
    lookup = {f["id"]: f for f in upright}
    while len(selected) < count:
        vectors = np.asarray([angular_feature(lookup[key]) for key in selected])
        best_id, best_score = None, -np.inf
        for frame in upright:
            if frame["id"] in selected:
                continue
            vec = angular_feature(frame)
            angles = np.arccos(np.clip(vectors @ vec, -1.0, 1.0))
            novelty = float(np.min(angles))
            elevation_bonus = 0.08 * math.sin(math.radians(float(frame["elevation_deg"])))
            score = novelty + elevation_bonus
            if score > best_score:
                best_score, best_id = score, frame["id"]
        selected.append(best_id)
    return [lookup[key] for key in selected]


def run_method(method: str, obs: Path, frames: list[dict], intrinsic, calib: dict,
               args: argparse.Namespace, out: Path) -> dict:
    start = time.perf_counter()
    voxel = args.voxel_mm / 1000.0
    details: dict = {}
    if method == "tsdf_fk":
        use = select_frames(frames, args.max_frames, include_flipped=False)
        mesh = integrate_tsdf(obs, use, intrinsic, voxel)
        details.update(frames=len(use), sensor="RGB-D", pose="reported FK/hand-eye")
    elif method == "tsdf_icp":
        use = select_frames(frames, args.max_frames, include_flipped=False)
        transforms, refinement = bounded_icp_transforms(obs, use, intrinsic, voxel)
        mesh = integrate_tsdf(obs, use, intrinsic, voxel, transforms)
        details.update(frames=len(use), sensor="RGB-D", pose="bounded ICP-refined FK/hand-eye", refinement=refinement)
    elif method == "poisson_depth":
        use = select_frames(frames, args.max_frames, include_flipped=False)
        points = fused_points(obs, use, intrinsic, voxel)
        mesh, extra = poisson_mesh(points, args.poisson_depth)
        details.update(frames=len(use), sensor="RGB-D", **extra)
    elif method == "laser_stripe_poisson":
        use = select_frames(frames, args.max_frames, include_flipped=False)
        points = fused_points(obs, use, intrinsic, max(voxel * 0.65, 0.00035), stripe_half_width=2)
        mesh, extra = poisson_mesh(points, args.poisson_depth)
        details.update(
            frames=len(use), sensor="idealized 5-pixel line-depth stripe",
            caveat="simulation analogue only; no physical laser profiler was available", **extra,
        )
    elif method == "visual_hull":
        use = select_frames(frames, args.max_frames, include_flipped=False)
        mesh, extra = visual_hull_mesh(obs, use, intrinsic, calib, args.visual_hull_mm / 1000.0)
        details.update(frames=len(use), sensor="RGB silhouettes plus depth-bounded volume", **extra)
    elif method == "nbv_tsdf":
        use = nbv_select(frames, args.nbv_views)
        mesh = integrate_tsdf(obs, use, intrinsic, voxel)
        details.update(
            frames=len(use), sensor="RGB-D", selected_ids=[f["id"] for f in use],
            caveat="pose-novelty/uncertainty proxy, not a full online OctoMap entropy controller",
        )
    elif method == "dual_pass_tsdf":
        use = select_frames(frames, args.max_frames, include_flipped=True)
        if not any(f["pass"] == "flipped" for f in use):
            raise RuntimeError("dual_pass_tsdf requires a capture created with --dual-pass")
        mesh = integrate_tsdf(obs, use, intrinsic, voxel)
        details.update(
            frames=len(use), sensor="RGB-D", passes=sorted(set(f["pass"] for f in use)),
            caveat="idealized regrasp transform is included as a measured observation",
        )
    elif method == "photometric_poisson":
        use = select_frames(frames, args.max_frames, include_flipped=False)
        points, photo = photometric_points(obs, use, intrinsic, calib, voxel)
        mesh, extra = poisson_mesh(points, args.poisson_depth)
        details.update(
            frames=len(use), sensor="four-light RGB normals plus noisy depth points",
            caveat="simulated calibrated lights; the real cell currently has no controllable ring light",
            **photo, **extra,
        )
    elif method == "touch_scaled_tsdf":
        touch_path = obs / "touch_probe.json"
        if not touch_path.exists():
            raise RuntimeError("touch_scaled_tsdf requires simulated/real touch_probe.json observations")
        touch = json.loads(touch_path.read_text(encoding="utf-8"))
        use = select_frames(frames, args.max_frames, include_flipped=False)
        mesh = integrate_tsdf(obs, use, intrinsic, voxel)
        xyz = np.asarray(mesh.vertices)
        before_lo, before_hi = xyz.min(axis=0), xyz.max(axis=0)
        target_lo = np.asarray(touch["bounds_m"][0], dtype=np.float64)
        target_hi = np.asarray(touch["bounds_m"][1], dtype=np.float64)
        scale = (target_hi - target_lo) / np.maximum(before_hi - before_lo, 1e-9)
        corrected = target_lo + (xyz - before_lo) * scale
        mesh.vertices = o3d.utility.Vector3dVector(corrected)
        mesh.compute_vertex_normals()
        details.update(
            frames=len(use), sensor="RGB-D plus six-axis simulated contact bounds",
            affine_scale_xyz=scale.tolist(), touch_noise_mm=touch.get("noise_sigma_mm"),
            caveat="sparse contact extent correction, not full ARAP bore/face probing",
        )
    else:
        raise ValueError(method)

    exported = export_mesh(mesh, out / method)
    return {
        "method": method,
        "status": "completed",
        "runtime_s": time.perf_counter() - start,
        **details,
        **exported,
    }


def main() -> int:
    args = parse_args()
    root = Path(args.dataset).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    obs = root / "observations"
    manifest = load_manifest(root)
    intrinsic, calib = intrinsics(manifest)
    frames = manifest["frames"]
    results = []
    for method in args.methods:
        print(f"[reconstruct] START {method}", flush=True)
        try:
            report = run_method(method, obs, frames, intrinsic, calib, args, out)
        except Exception as exc:
            report = {"method": method, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        (out / f"{method}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        results.append(report)
        print(f"[reconstruct] {report['status'].upper()} {method}", flush=True)
    summary = {"dataset": str(root), "source_mesh_access": False, "methods": results}
    (out / "reconstruction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if any(r["status"] == "completed" for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
