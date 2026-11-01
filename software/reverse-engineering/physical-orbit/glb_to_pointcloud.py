"""Extract the `points` node from a DVLT-style GLB into a PLY point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    loaded = trimesh.load(args.input, force='scene')
    candidates = []
    for node_name in loaded.graph.nodes_geometry:
        transform, geom_name = loaded.graph.get(node_name)
        geom = loaded.geometry[geom_name]
        if not hasattr(geom, 'vertices'):
            continue
        vertices = trimesh.transform_points(np.asarray(geom.vertices), transform)
        colors = getattr(getattr(geom, 'visual', None), 'vertex_colors', None)
        if colors is None and hasattr(geom, 'colors'):
            colors = geom.colors
        candidates.append((node_name, geom_name, vertices, colors))
    if not candidates:
        raise RuntimeError(f'No vertex geometry in {args.input}')
    preferred = [entry for entry in candidates if entry[0] == 'points' or entry[1] == 'points']
    chosen = max(preferred or candidates, key=lambda entry: len(entry[2]))
    node_name, geom_name, vertices, colors = chosen
    if colors is None:
        colors = np.full((len(vertices), 4), 255, dtype=np.uint8)
    colors = np.asarray(colors)
    if colors.ndim == 1:
        colors = np.tile(colors, (len(vertices), 1))
    colors = colors[:, :4].astype(np.uint8)
    cloud = trimesh.PointCloud(vertices=vertices, colors=colors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cloud.export(args.output)
    manifest = {
        'input': str(args.input), 'output': str(args.output), 'node': node_name,
        'geometry': geom_name, 'points': int(len(vertices)),
        'available_nodes': [{'node': n, 'geometry': g, 'points': int(len(v))} for n, g, v, _ in candidates],
    }
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
