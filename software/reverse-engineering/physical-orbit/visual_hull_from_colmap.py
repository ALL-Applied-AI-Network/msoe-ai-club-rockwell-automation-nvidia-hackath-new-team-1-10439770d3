"""Carve an RGB silhouette visual hull using a COLMAP camera solution.

The only object cue is the frozen pre-evaluation mask set.  The evaluator CAD
is never read.  Requiring a configurable view consensus makes the method robust
to a few mask or camera-pose errors while retaining a strict multi-view hull.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from skimage.measure import marching_cubes
import trimesh


def read_camera(path: Path):
    for line in path.read_text().splitlines():
        if line and not line.startswith('#'):
            fields = line.split()
            model = fields[1]
            if model != 'SIMPLE_RADIAL':
                raise ValueError(f'Expected SIMPLE_RADIAL, got {model}')
            width, height = int(fields[2]), int(fields[3])
            f, cx, cy, k = map(float, fields[4:8])
            return width, height, f, cx, cy, k
    raise RuntimeError(f'No camera in {path}')


def read_images(path: Path):
    output = []
    expect_header = True
    for line in path.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        if expect_header:
            fields = line.split()
            image_id = int(fields[0])
            qw, qx, qy, qz = map(float, fields[1:5])
            t = np.asarray(list(map(float, fields[5:8])), dtype=np.float64)
            name = fields[9]
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            output.append({'id': image_id, 'name': name, 'R': R, 't': t})
        expect_header = not expect_header
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-txt', required=True, type=Path)
    parser.add_argument('--masks', required=True, type=Path)
    parser.add_argument('--bounds-cloud', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--min-views', type=int, default=16)
    parser.add_argument('--mask-dilate', type=int, default=2)
    parser.add_argument('--bounds-margin', type=float, default=0.25)
    parser.add_argument('--table-cap', action='store_true', help='Close the unseen bottom at the observed lower-Z support plane.')
    parser.add_argument('--depth-z-cap', action='store_true', help='Bound the upper surface by the robust MVS depth-cloud Z extent.')
    parser.add_argument('--chunk', type=int, default=500_000)
    args = parser.parse_args()

    width, height, f, cx, cy, k = read_camera(args.model_txt / 'cameras.txt')
    cameras = read_images(args.model_txt / 'images.txt')
    if not (1 <= args.min_views <= len(cameras)):
        raise ValueError('min-views must be between 1 and the camera count')

    loaded = trimesh.load(args.bounds_cloud, process=False)
    points = np.asarray(loaded.vertices, dtype=np.float64)
    lower = np.percentile(points, 0.25, axis=0)
    upper = np.percentile(points, 99.75, axis=0)
    span = upper - lower
    observed_lower_z = float(lower[2])
    observed_upper_z = float(upper[2])
    lower -= args.bounds_margin * span
    upper += args.bounds_margin * span
    if args.table_cap:
        lower[2] = observed_lower_z - 0.03 * span[2]
    if args.depth_z_cap:
        upper[2] = observed_upper_z + 0.03 * span[2]

    masks = []
    kernel = np.ones((3, 3), np.uint8)
    for camera in cameras:
        mask_path = args.masks / camera['name']
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)
        if args.mask_dilate:
            mask = cv2.dilate(mask, kernel, iterations=args.mask_dilate)
        masks.append(mask.astype(bool))

    n = args.resolution
    shape = (n, n, n)
    steps = (upper - lower) / (np.asarray(shape) - 1)
    total = n ** 3
    occupied = np.zeros(total, dtype=bool)
    for start in range(0, total, args.chunk):
        stop = min(total, start + args.chunk)
        idx = np.arange(start, stop, dtype=np.int64)
        ix = idx // (n * n)
        rem = idx % (n * n)
        iy = rem // n
        iz = rem % n
        world = lower + np.column_stack((ix, iy, iz)) * steps
        support = np.zeros(len(world), dtype=np.uint8)
        for camera, mask in zip(cameras, masks):
            cam = world @ camera['R'].T + camera['t']
            z = cam[:, 2]
            safe_z = np.where(z > 1e-9, z, 1.0)
            xn, yn = cam[:, 0] / safe_z, cam[:, 1] / safe_z
            radial = 1.0 + k * (xn * xn + yn * yn)
            u = np.rint(f * xn * radial + cx).astype(np.int32)
            v = np.rint(f * yn * radial + cy).astype(np.int32)
            inside = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            hit = np.zeros(len(world), dtype=bool)
            hit[inside] = mask[v[inside], u[inside]]
            support += hit
        occupied[start:stop] = support >= args.min_views

    volume = occupied.reshape(shape)
    if args.table_cap:
        # The orbit is upper-hemisphere only, so the bottom is physically
        # unobserved. An empty boundary layer turns the table support plane
        # into an explicit, auditable planar cap instead of pretending it was
        # optically reconstructed.
        volume[:, :, 0] = False
    if args.depth_z_cap:
        volume[:, :, -1] = False
    boundary_counts = {
        'x_min': int(volume[0, :, :].sum()), 'x_max': int(volume[-1, :, :].sum()),
        'y_min': int(volume[:, 0, :].sum()), 'y_max': int(volume[:, -1, :].sum()),
        'z_min': int(volume[:, :, 0].sum()), 'z_max': int(volume[:, :, -1].sum()),
    }
    if not volume.any():
        raise RuntimeError('Visual hull is empty; lower min-views or increase mask dilation')
    vertices, faces, normals, _ = marching_cubes(volume.astype(np.float32), 0.5, spacing=steps)
    vertices += lower
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=True)
    components = mesh.split(only_watertight=False)
    if components:
        mesh = max(components, key=lambda item: abs(item.volume) if item.is_watertight else item.area)
    mesh.remove_unreferenced_vertices()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)
    manifest = {
        'output': str(args.output), 'cameras': len(cameras), 'resolution': n,
        'min_views': args.min_views, 'mask_dilate': args.mask_dilate,
        'bounds_margin': args.bounds_margin,
        'bottom_inferred_from_table_support': bool(args.table_cap),
        'upper_z_bounded_by_mvs_depth': bool(args.depth_z_cap),
        'bounds_min': lower.tolist(), 'bounds_max': upper.tolist(),
        'occupied_voxels': int(volume.sum()), 'vertices': int(len(mesh.vertices)),
        'occupied_boundary_voxels': boundary_counts,
        'faces': int(len(mesh.faces)), 'watertight': bool(mesh.is_watertight),
        'components_before_largest_selection': len(components),
        'cad_used': False,
    }
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
