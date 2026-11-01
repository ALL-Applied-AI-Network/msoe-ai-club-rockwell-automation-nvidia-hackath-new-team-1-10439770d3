"""Build and render a CAD-derived idealized photogrammetry reference.

The output is intentionally and visibly labelled as synthetic.  It answers a
specific engineering question: *what should a strong RGB orbit reconstruction
of this part look like if capture coverage and registration are good?*

The program creates:

* an exact colored CAD assembly;
* a dense point/surfel cloud with bounded view-registration and range noise;
* a watertight, near-functional surface reference with sub-0.1 mm perturbation;
* 4K/1080p stills, diagnostic views, and smooth turntable videos.

No image-to-geometry method or evaluator CAD is involved.  The two input 3MF
files are the sole geometry source.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import moderngl
import numpy as np
import open3d as o3d
import trimesh


BASE_RGB = np.array([28, 92, 205], dtype=np.float32) / 255.0
GEAR_RGB = np.array([43, 218, 116], dtype=np.float32) / 255.0
BACKGROUND_BGR = np.array([13, 18, 25], dtype=np.uint8)
SEED = 20261027


def load_mesh(path: Path) -> trimesh.Trimesh:
    # Preserve the author's exact indexed topology.  Trimesh's generic repair
    # pass merges a handful of deliberately coincident FDM features in this
    # 3MF and incorrectly turns the watertight base into multiple shells.
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    loaded.remove_unreferenced_vertices()
    loaded.fix_normals()
    return loaded


def color_mesh(mesh: trimesh.Trimesh, rgb: np.ndarray) -> trimesh.Trimesh:
    result = mesh.copy()
    rgba = np.rint(np.r_[rgb * 255.0, 255.0]).astype(np.uint8)
    result.visual.face_colors = np.tile(rgba, (len(result.faces), 1))
    result.visual.vertex_colors = np.tile(rgba, (len(result.vertices), 1))
    return result


def rotation_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    rxm = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rym = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rzm = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rzm @ rym @ rxm


def make_assembly(base_path: Path, gear_path: Path) -> tuple[trimesh.Trimesh, trimesh.Trimesh, trimesh.Trimesh]:
    base = color_mesh(load_mesh(base_path), BASE_RGB)
    gear = color_mesh(load_mesh(gear_path), GEAR_RGB)
    # Measured from the shaft centers and thrust surface in the supplied base.
    gear.apply_translation([-95.0, 0.0, 8.0])
    assembly = trimesh.util.concatenate((base, gear))
    assembly.fix_normals()
    return base, gear, assembly


def sample_component(
    mesh: trimesh.Trimesh,
    count: int,
    base_rgb: np.ndarray,
    view_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points, face_ids = trimesh.sample.sample_surface(mesh, count)
    normals = mesh.face_normals[face_ids].copy()

    # Virtual upper-hemisphere wrist orbit.  Each surface sample is assigned to
    # the most compatible view so grazing-angle uncertainty can be modelled.
    az = np.linspace(0.0, 2.0 * np.pi, view_count, endpoint=False)
    elevations = np.deg2rad(32.0 + 14.0 * np.sin(az * 2.0 + 0.4))
    view_dirs = np.column_stack(
        [np.cos(elevations) * np.cos(az), np.cos(elevations) * np.sin(az), np.sin(elevations)]
    )
    facing = normals @ view_dirs.T
    best_view = np.argmax(facing, axis=1)
    confidence = np.clip(np.max(facing, axis=1), 0.0, 1.0)

    # An upper orbit cannot measure the underside.  Keep a tiny amount near the
    # silhouette to mimic fused edge rays, but reject true downward surfaces.
    coverage_probability = np.where(
        normals[:, 2] < -0.35,
        0.015,
        0.18 + 0.82 * np.power(confidence, 0.65),
    )
    keep = rng.random(count) < coverage_probability
    points, normals = points[keep], normals[keep]
    confidence, best_view = confidence[keep], best_view[keep]

    # Small per-view rigid registration residuals.  These are deliberately
    # below the part's functional clearances and much smaller than the failed
    # real COLMAP cloud, representing a strong future capture.
    rotations = []
    translations = []
    for _ in range(view_count):
        rotations.append(rotation_matrix_xyz(*np.deg2rad(rng.normal(0.0, 0.025, 3))))
        translations.append(rng.normal(0.0, 0.055, 3))
    rotations = np.stack(rotations)
    translations = np.stack(translations)
    transformed = np.empty_like(points)
    transformed_normals = np.empty_like(normals)
    for view_id in range(view_count):
        mask = best_view == view_id
        if not np.any(mask):
            continue
        transformed[mask] = points[mask] @ rotations[view_id].T + translations[view_id]
        transformed_normals[mask] = normals[mask] @ rotations[view_id].T

    # Range noise rises at grazing incidence.  Tangential noise remains small,
    # preserving edges while leaving a credible photogrammetric surface grain.
    sigma_normal = 0.025 + 0.095 * np.square(1.0 - confidence)
    normal_noise = rng.normal(0.0, sigma_normal)[:, None] * transformed_normals
    tangent_noise = rng.normal(0.0, 0.018, transformed.shape)
    tangent_noise -= (tangent_noise * transformed_normals).sum(axis=1)[:, None] * transformed_normals
    transformed += normal_noise + tangent_noise

    exposure = rng.normal(1.0, 0.025, view_count)[best_view]
    color = base_rgb[None, :] * exposure[:, None] + rng.normal(0.0, 0.010, transformed.shape)
    color = np.clip(color, 0.0, 1.0)
    return transformed, transformed_normals, color, confidence, best_view


def make_scan_cloud(
    base: trimesh.Trimesh,
    gear: trimesh.Trimesh,
    total_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    base_count = int(total_points * 0.76)
    gear_count = total_points - base_count
    parts = [
        sample_component(base, base_count, BASE_RGB, 48, rng),
        sample_component(gear, gear_count, GEAR_RGB, 48, rng),
    ]
    return tuple(np.concatenate([part[index] for part in parts], axis=0) for index in range(5))  # type: ignore[return-value]


def export_point_cloud(path: Path, points: np.ndarray, normals: np.ndarray, colors: np.ndarray) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.normals = o3d.utility.Vector3dVector(normals)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False)


def make_functional_surface(base: trimesh.Trimesh, gear: trimesh.Trimesh) -> trimesh.Trimesh:
    rng = np.random.default_rng(SEED + 17)
    outputs = []
    for source in (base, gear):
        mesh = source.copy()
        mesh.vertex_normals  # force cache
        vertices = mesh.vertices.copy()
        normals = mesh.vertex_normals.copy()
        low_frequency = (
            0.022 * np.sin(vertices[:, 0] * 0.071)
            + 0.018 * np.sin(vertices[:, 1] * 0.093 + 0.7)
            + 0.012 * np.sin(vertices[:, 2] * 0.16 + 1.1)
        )
        displacement = np.clip(low_frequency + rng.normal(0.0, 0.014, len(vertices)), -0.075, 0.075)
        mesh.vertices = vertices + normals * displacement[:, None]
        mesh.fix_normals()
        outputs.append(mesh)
    return trimesh.util.concatenate(outputs)


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = normalize(target - eye)
    side = normalize(np.cross(forward, up))
    camera_up = np.cross(side, forward)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, :3] = side
    matrix[1, :3] = camera_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -np.dot(side, eye)
    matrix[1, 3] = -np.dot(camera_up, eye)
    matrix[2, 3] = np.dot(forward, eye)
    return matrix


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2.0)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


POINT_VERTEX = """
#version 330
uniform mat4 mvp;
uniform mat4 view;
uniform vec3 light_dir;
uniform float point_scale;
in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;
in float in_confidence;
out vec3 v_color;
out float v_light;
out float v_confidence;
void main() {
    vec4 camera = view * vec4(in_position, 1.0);
    gl_Position = mvp * vec4(in_position, 1.0);
    gl_PointSize = clamp(point_scale / max(0.45, -camera.z), 2.0, 5.2);
    vec3 n = normalize(in_normal);
    v_light = 0.58 + 0.42 * max(0.0, dot(n, normalize(light_dir)));
    v_color = in_color;
    v_confidence = in_confidence;
}
"""


POINT_FRAGMENT = """
#version 330
in vec3 v_color;
in float v_light;
in float v_confidence;
out vec4 fragColor;
void main() {
    vec2 q = gl_PointCoord - vec2(0.5);
    float radius = length(q);
    if (radius > 0.5) discard;
    float alpha = smoothstep(0.50, 0.34, radius);
    float confidence_tint = 0.88 + 0.12 * v_confidence;
    fragColor = vec4(v_color * v_light * confidence_tint, alpha);
}
"""


MESH_VERTEX = """
#version 330
uniform mat4 mvp;
uniform mat4 view;
uniform vec3 light_dir;
in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;
out vec3 v_color;
out float v_light;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    vec3 n = normalize(in_normal);
    v_light = 0.48 + 0.52 * max(0.0, dot(n, normalize(light_dir)));
    v_color = in_color;
}
"""


MESH_FRAGMENT = """
#version 330
in vec3 v_color;
in float v_light;
out vec4 fragColor;
void main() {
    vec3 color = v_color * v_light;
    fragColor = vec4(color, 1.0);
}
"""


class ReferenceRenderer:
    def __init__(
        self,
        width: int,
        height: int,
        points: np.ndarray,
        normals: np.ndarray,
        colors: np.ndarray,
        confidence: np.ndarray,
        surface: trimesh.Trimesh,
    ) -> None:
        self.width = width
        self.height = height
        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.color_texture = self.ctx.texture((width, height), 4, dtype="f1")
        self.depth_texture = self.ctx.depth_texture((width, height))
        self.framebuffer = self.ctx.framebuffer(self.color_texture, self.depth_texture)

        point_data = np.column_stack((points, normals, colors, confidence)).astype("f4")
        self.point_program = self.ctx.program(vertex_shader=POINT_VERTEX, fragment_shader=POINT_FRAGMENT)
        self.point_vbo = self.ctx.buffer(point_data.tobytes())
        self.point_vao = self.ctx.vertex_array(
            self.point_program,
            [(self.point_vbo, "3f 3f 3f 1f", "in_position", "in_normal", "in_color", "in_confidence")],
        )

        mesh = surface.copy()
        mesh.unmerge_vertices()
        vertex_colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
        mesh_data = np.column_stack((mesh.vertices, mesh.vertex_normals, vertex_colors)).astype("f4")
        self.mesh_program = self.ctx.program(vertex_shader=MESH_VERTEX, fragment_shader=MESH_FRAGMENT)
        self.mesh_vbo = self.ctx.buffer(mesh_data.tobytes())
        self.mesh_vao = self.ctx.vertex_array(
            self.mesh_program,
            [(self.mesh_vbo, "3f 3f 3f", "in_position", "in_normal", "in_color")],
        )

        bounds = np.vstack((points.min(axis=0), points.max(axis=0)))
        self.center = bounds.mean(axis=0)
        self.center[2] += 2.0
        self.span = bounds[1] - bounds[0]

    def render(self, azimuth_deg: float, elevation_deg: float, mode: str = "points") -> np.ndarray:
        distance = max(self.span[0] * 1.58, self.span[1] * 2.7)
        azimuth = math.radians(azimuth_deg)
        elevation = math.radians(elevation_deg)
        eye = self.center + distance * np.array(
            [math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)]
        )
        view = look_at(eye.astype(np.float32), self.center.astype(np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32))
        projection = perspective(31.0, self.width / self.height, distance * 0.12, distance * 3.2)
        mvp = projection @ view

        self.framebuffer.use()
        self.framebuffer.clear(0.035, 0.052, 0.073, 1.0, depth=1.0)
        light = normalize(np.array([-0.35, -0.25, 0.9], dtype=np.float32))
        if mode == "mesh":
            program, vao = self.mesh_program, self.mesh_vao
            program["mvp"].write(mvp.T.astype("f4").tobytes())
            program["light_dir"].value = tuple(light.tolist())
            vao.render(mode=moderngl.TRIANGLES)
        else:
            program, vao = self.point_program, self.point_vao
            program["mvp"].write(mvp.T.astype("f4").tobytes())
            program["view"].write(view.T.astype("f4").tobytes())
            program["light_dir"].value = tuple(light.tolist())
            program["point_scale"].value = 1450.0 if self.height >= 1000 else 980.0
            vao.render(mode=moderngl.POINTS)

        data = self.framebuffer.read(components=3, alignment=1)
        rgb = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)[::-1]
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def add_polish(
    image: np.ndarray,
    title: str,
    subtitle: str,
    azimuth: float,
    mode: str,
    footer_left: str = "48-view upper orbit | RGB-derived geometry target | sub-0.1 mm synthetic residual",
) -> np.ndarray:
    height, width = image.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    radial = ((xx - width * 0.5) / (width * 0.72)) ** 2 + ((yy - height * 0.48) / (height * 0.85)) ** 2
    vignette = 1.0 - 0.20 * np.clip(radial, 0.0, 1.0)
    image = np.clip(image.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)

    # Quiet, engineering-style hierarchy; no arrows or promotional language.
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, int(height * 0.135)), (8, 13, 19), -1)
    cv2.rectangle(overlay, (0, int(height * 0.88)), (width, height), (8, 13, 19), -1)
    image = cv2.addWeighted(overlay, 0.88, image, 0.12, 0)
    scale = width / 1920.0
    cv2.putText(image, title, (int(54 * scale), int(60 * scale)), cv2.FONT_HERSHEY_DUPLEX, 1.14 * scale, (240, 244, 248), 2, cv2.LINE_AA)
    cv2.putText(image, subtitle, (int(56 * scale), int(99 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.57 * scale, (150, 168, 185), 1, cv2.LINE_AA)
    badge = "CAD-SYNTHETIC REFERENCE"
    badge_size = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.50 * scale, 1)[0]
    bx = width - badge_size[0] - int(58 * scale)
    by = int(75 * scale)
    cv2.rectangle(image, (bx - int(14 * scale), by - badge_size[1] - int(10 * scale)), (width - int(42 * scale), by + int(9 * scale)), (37, 50, 63), -1)
    cv2.putText(image, badge, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.50 * scale, (199, 211, 222), 1, cv2.LINE_AA)

    footer_y = height - int(44 * scale)
    left = footer_left
    right = f"{mode.upper()}  /  AZ {azimuth % 360:05.1f} DEG"
    cv2.putText(image, left, (int(56 * scale), footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.49 * scale, (135, 154, 172), 1, cv2.LINE_AA)
    right_size = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.49 * scale, 1)[0]
    cv2.putText(image, right, (width - right_size[0] - int(56 * scale), footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.49 * scale, (135, 154, 172), 1, cv2.LINE_AA)
    return image


def write_video(
    renderer: ReferenceRenderer,
    path: Path,
    mode: str,
    frames: int,
    fps: float,
    title: str,
) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (renderer.width, renderer.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {path}")
    for index in range(frames):
        phase = index / frames
        azimuth = 220.0 + phase * 360.0
        elevation = 29.0 + 5.0 * math.sin(phase * 2.0 * math.pi)
        frame = renderer.render(azimuth, elevation, mode=mode)
        frame = add_polish(
            frame,
            title,
            "Reference appearance for a well-registered wrist-camera orbit",
            azimuth,
            mode,
        )
        writer.write(frame)
    writer.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--gear", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=1_200_000)
    parser.add_argument("--frames", type=int, default=180)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    base, gear, exact = make_assembly(args.base, args.gear)
    points, normals, colors, confidence, view_ids = make_scan_cloud(base, gear, args.points)
    functional = make_functional_surface(base, gear)

    exact.export(args.output / "exact_cad_assembly.glb")
    exact.export(args.output / "exact_cad_assembly.ply")
    functional.export(args.output / "ideal_functional_reconstruction.obj")
    functional.export(args.output / "ideal_functional_reconstruction.stl")
    functional.export(args.output / "ideal_functional_reconstruction.ply")
    export_point_cloud(args.output / "ideal_registered_scan.ply", points, normals, colors)

    report = {
        "source": {
            "base": str(args.base.resolve()),
            "gear": str(args.gear.resolve()),
            "cad_only": True,
            "real_images_used_for_geometry": False,
        },
        "assembly": {
            "gear_translation_mm": [-95.0, 0.0, 8.0],
            "bounds_mm": exact.bounds.tolist(),
            "extents_mm": exact.extents.tolist(),
            "exact_watertight": bool(exact.is_watertight),
            "functional_surface_watertight": bool(functional.is_watertight),
        },
        "ideal_scan_model": {
            "points": int(len(points)),
            "virtual_views": 48,
            "registration_rotation_sigma_deg": 0.025,
            "registration_translation_sigma_mm": 0.055,
            "range_noise_sigma_mm": [0.025, 0.120],
            "confidence_mean": float(np.mean(confidence)),
            "confidence_p05": float(np.percentile(confidence, 5)),
            "confidence_p95": float(np.percentile(confidence, 95)),
            "views_contributing": int(len(np.unique(view_ids))),
            "bottom_face_observed": False,
        },
        "interpretation": "Target appearance only. It is generated from CAD and is not evidence of real scan accuracy.",
    }
    (args.output / "reference_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    renderer = ReferenceRenderer(1920, 1080, points, normals, colors, confidence, functional)
    hero = add_polish(
        renderer.render(222.0, 33.0, mode="points"),
        "Idealized RGB reconstruction",
        "Dense registered surfels from the supplied gearbox CAD",
        222.0,
        "points",
    )
    cv2.imwrite(str(args.output / "ideal_reconstruction_hero.png"), hero)

    views = []
    for azimuth in (210.0, 300.0, 30.0, 120.0):
        panel = renderer.render(azimuth, 34.0, mode="points")
        panel = cv2.resize(panel, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (22, 474), (198, 520), (8, 13, 19), -1)
        cv2.putText(panel, f"AZ {azimuth:03.0f} DEG", (34, 506), cv2.FONT_HERSHEY_DUPLEX, 0.68, (220, 230, 238), 1, cv2.LINE_AA)
        views.append(panel)
    sheet = np.vstack((np.hstack(views[:2]), np.hstack(views[2:])))
    cv2.rectangle(sheet, (0, 0), (sheet.shape[1], 82), (8, 13, 19), -1)
    cv2.putText(sheet, "Ideal orbit coverage - four registered viewpoints", (44, 53), cv2.FONT_HERSHEY_DUPLEX, 1.05, (240, 244, 248), 2, cv2.LINE_AA)
    cv2.putText(sheet, "CAD-SYNTHETIC / NOT A MEASURED SCAN", (1420, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (163, 180, 195), 1, cv2.LINE_AA)
    cv2.imwrite(str(args.output / "ideal_reconstruction_four_views.png"), sheet)

    top_view = add_polish(
        renderer.render(270.0, 68.0, mode="points"),
        "Idealized reconstruction - high-angle view",
        "Feature separation expected from complete upper-hemisphere coverage",
        270.0,
        "points",
    )
    cv2.imwrite(str(args.output / "ideal_reconstruction_high_angle.png"), top_view)

    surface = add_polish(
        renderer.render(222.0, 33.0, mode="mesh"),
        "Near-functional surface reference",
        "Watertight CAD topology with bounded sub-0.1 mm surface residual",
        222.0,
        "mesh",
    )
    cv2.imwrite(str(args.output / "ideal_surface_hero.png"), surface)

    write_video(renderer, args.output / "ideal_reconstruction_turntable.mp4", "points", args.frames, 30.0, "Idealized RGB reconstruction")
    write_video(renderer, args.output / "ideal_surface_turntable.mp4", "mesh", args.frames, 30.0, "Near-functional surface reference")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
