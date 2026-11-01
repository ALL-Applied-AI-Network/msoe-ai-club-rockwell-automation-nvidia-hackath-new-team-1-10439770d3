"""Export BLASt3R .pth depth maps as masked colored point clouds.

This adapter intentionally consumes only candidate-produced camera/depth/confidence
and the masks frozen before evaluation.  It never reads the evaluator CAD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.squeeze(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        if image.max(initial=0) <= 1.5:
            image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def find_mask(mask_dir: Path, name: str, shape: tuple[int, int]) -> np.ndarray:
    stem = Path(name).stem
    matches = list(mask_dir.glob(stem + '.*'))
    if not matches:
        matches = list(mask_dir.glob('*' + stem + '*'))
    if not matches:
        raise FileNotFoundError(f'No frozen mask for view {name!r} under {mask_dir}')
    mask = cv2.imread(str(matches[0]), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f'Could not read mask {matches[0]}')
    mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode((mask > 127).astype(np.uint8), kernel, iterations=2).astype(bool)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, type=Path)
    parser.add_argument('--masks', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--confidence-percentile', type=float, default=25.0)
    parser.add_argument('--max-points', type=int, default=1_500_000)
    args = parser.parse_args()

    scene = torch.load(args.scene, map_location='cpu', weights_only=False)
    views = scene.get('views', scene)
    if not isinstance(views, dict):
        raise TypeError(f'Expected scene["views"] mapping, got {type(views).__name__}')

    all_points, all_colors, per_view = [], [], []
    for view_name, view in views.items():
        depth = np.squeeze(as_numpy(view['depth'])).astype(np.float64)
        conf = np.squeeze(as_numpy(view.get('conf', np.ones_like(depth)))).astype(np.float64)
        K = np.squeeze(as_numpy(view['K'])).astype(np.float64)
        cam2w = np.squeeze(as_numpy(view['cam2w'])).astype(np.float64)
        image = normalize_image(as_numpy(view['img']))
        image = cv2.resize(image, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
        mask = find_mask(args.masks, str(view_name), depth.shape)

        valid = mask & np.isfinite(depth) & (depth > 0) & np.isfinite(conf)
        threshold = None
        if valid.any() and args.confidence_percentile > 0:
            threshold = float(np.percentile(conf[valid], args.confidence_percentile))
            valid &= conf >= threshold
        vv, uu = np.nonzero(valid)
        z = depth[vv, uu]
        x = (uu - K[0, 2]) * z / K[0, 0]
        y = (vv - K[1, 2]) * z / K[1, 1]
        cam = np.column_stack((x, y, z, np.ones_like(z)))
        world = (cam2w @ cam.T).T[:, :3]
        colors = image[vv, uu].astype(np.float64) / 255.0
        all_points.append(world)
        all_colors.append(colors)
        per_view.append({'view': str(view_name), 'points': int(len(world)), 'confidence_threshold': threshold})

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    rng = np.random.default_rng(20261027)
    if len(points) > args.max_points:
        keep = rng.choice(len(points), size=args.max_points, replace=False)
        points, colors = points[keep], colors[keep]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.empty(
        len(points),
        dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')],
    )
    vertices['x'], vertices['y'], vertices['z'] = points[:, 0], points[:, 1], points[:, 2]
    rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    vertices['red'], vertices['green'], vertices['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        'ply\nformat binary_little_endian 1.0\n'
        f'element vertex {len(vertices)}\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
    ).encode('ascii')
    with args.output.open('wb') as handle:
        handle.write(header)
        vertices.tofile(handle)
    manifest = {
        'scene': str(args.scene),
        'masks': str(args.masks),
        'output': str(args.output),
        'points': int(len(points)),
        'confidence_percentile': args.confidence_percentile,
        'per_view': per_view,
        'note': 'Only frozen RGB-derived masks and candidate-produced depth/cameras/confidence were used.',
    }
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
