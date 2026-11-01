#!/usr/bin/env python3
"""Export exact, assembly-coordinate inline-four engine assets from 3MF.

The source OpenSCAD exports were saved in exploded view.  This converter
reverses those authored Z translations, removes four zero-area cover slivers,
splits the rod and piston groups into individually placeable components, and
emits twelve binary STL files in one shared final-assembly coordinate frame.
STL has no material channel, so the native 3MF material name and RGB value are
preserved in the manifest used by the Isaac renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import trimesh


GROUPS = {
    "block": {
        "source": "block.3mf",
        "z_correction_mm": 0.0,
        "colour_name": "SteelBlue",
        "display_colour": "#4682B4",
        "split_count": 1,
    },
    "crankshaft": {
        "source": "crankshaft.3mf",
        "z_correction_mm": -30.0,
        "colour_name": "Silver",
        "display_colour": "#C0C0C0",
        "split_count": 1,
    },
    "cover": {
        "source": "droponcover.3mf",
        "z_correction_mm": -60.0,
        "colour_name": "DimGray",
        "display_colour": "#696969",
        "split_count": 1,
    },
    "head": {
        "source": "cylinderhead.3mf",
        "z_correction_mm": -90.0,
        "colour_name": "Firebrick",
        "display_colour": "#B22222",
        "split_count": 1,
    },
    "rod": {
        "source": "rodsx4.3mf",
        "z_correction_mm": -145.3,
        "colour_name": "DarkOrange",
        "display_colour": "#FF8C00",
        "split_count": 4,
    },
    "piston": {
        "source": "pistonsx4.3mf",
        "z_correction_mm": -205.3,
        "colour_name": "Gainsboro",
        "display_colour": "#DCDCDC",
        "split_count": 4,
    },
}

CYLINDER_X_MM = [25.15, 1.15, -22.85, -46.85]
ASSEMBLY_ORDER = [
    "block",
    "crankshaft",
    "rod_1",
    "rod_2",
    "rod_3",
    "rod_4",
    "cover",
    "head",
    "piston_1",
    "piston_2",
    "piston_3",
    "piston_4",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_mesh(loaded: Any) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()
    if not isinstance(loaded, trimesh.Scene):
        raise TypeError(f"Unsupported trimesh result: {type(loaded)!r}")
    meshes = []
    for node_name in loaded.graph.nodes_geometry:
        transform, geometry_name = loaded.graph[node_name]
        mesh = loaded.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    if not meshes:
        raise ValueError("3MF scene contained no mesh geometry")
    return trimesh.util.concatenate(meshes)


def clean_for_components(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Merge duplicate vertices without repairing or decimating geometry."""
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64).copy(),
        faces=np.asarray(mesh.faces, dtype=np.int64).copy(),
        process=True,
        validate=False,
    )


def rounded(values: np.ndarray, digits: int = 6) -> list:
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def component_records(
    group_name: str,
    mesh: trimesh.Trimesh,
    expected_count: int,
) -> tuple[list[tuple[str, trimesh.Trimesh]], dict[str, Any]]:
    components = list(clean_for_components(mesh).split(only_watertight=False))
    slivers = [component for component in components if len(component.faces) <= 2 or abs(component.volume) < 1e-9]
    components = [component for component in components if len(component.faces) > 2 and abs(component.volume) >= 1e-9]
    if len(components) != expected_count:
        raise ValueError(
            f"{group_name}: expected {expected_count} substantive components, got {len(components)}; "
            f"discarded {len(slivers)} slivers"
        )

    if expected_count == 1:
        named = [(group_name, components[0])]
    else:
        # Cylinder identity is the nearest known final X coordinate, not an
        # incidental component ordering returned by trimesh.
        remaining = list(components)
        named = []
        for cylinder_index, target_x in enumerate(CYLINDER_X_MM, start=1):
            match = min(remaining, key=lambda item: abs(float(item.centroid[0]) - target_x))
            if abs(float(match.centroid[0]) - target_x) > 0.05:
                raise ValueError(
                    f"{group_name}: component for cylinder {cylinder_index} has unexpected "
                    f"X centroid {match.centroid[0]:.6f} mm"
                )
            remaining.remove(match)
            named.append((f"{group_name}_{cylinder_index}", match))

    audit = {
        "raw_connected_components": int(len(components) + len(slivers)),
        "discarded_sliver_components": int(len(slivers)),
        "discarded_sliver_triangles": int(sum(len(item.faces) for item in slivers)),
        "substantive_components": int(len(components)),
    }
    return named, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "cad" / "engine",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "assets",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    reference_path = source / "assembleed.3mf"
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    reference = clean_for_components(as_mesh(trimesh.load(reference_path, force="scene", process=False)))
    reference_tree = cKDTree(np.asarray(reference.vertices, dtype=float))

    manifest: dict[str, Any] = {
        "schema": "ateam.inline_four_engine_assets.v1",
        "units": "millimetres",
        "coordinate_frame": "shared final assembly frame from supplied OpenSCAD model",
        "source_directory": str(source),
        "assembled_reference_file": reference_path.name,
        "assembled_reference_sha256": sha256(reference_path),
        "assembled_reference_bounds_mm": rounded(reference.bounds),
        "assembled_reference_extents_mm": rounded(reference.extents),
        "assembly_order": ASSEMBLY_ORDER,
        "cylinder_x_mm": CYLINDER_X_MM,
        "groups": {},
        "parts": {},
    }

    exported_names: set[str] = set()
    for group_name, spec in GROUPS.items():
        source_path = source / spec["source"]
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        raw_mesh = as_mesh(trimesh.load(source_path, force="scene", process=False))
        correction = np.array((0.0, 0.0, float(spec["z_correction_mm"])))
        raw_mesh.apply_translation(correction)
        named_components, group_audit = component_records(
            group_name,
            raw_mesh,
            int(spec["split_count"]),
        )
        manifest["groups"][group_name] = {
            "source_file": source_path.name,
            "source_sha256": sha256(source_path),
            "exploded_to_assembly_translation_mm": correction.tolist(),
            "native_material_name": spec["colour_name"],
            "native_display_colour": spec["display_colour"],
            **group_audit,
        }

        for part_name, part_mesh in named_components:
            if part_name in exported_names:
                raise ValueError(f"Duplicate part name: {part_name}")
            exported_names.add(part_name)
            part_mesh.remove_unreferenced_vertices()
            if not part_mesh.is_watertight:
                raise ValueError(f"{part_name}: exported component is not watertight")
            output_path = output / f"{part_name}.stl"
            output_path.write_bytes(trimesh.exchange.stl.export_stl(part_mesh))
            nearest, _ = reference_tree.query(np.asarray(part_mesh.vertices), k=1)
            crosscheck = {
                "vertices_within_0_05_mm_fraction": round(float(np.mean(nearest <= 0.05)), 9),
                "median_nearest_reference_error_mm": round(float(np.median(nearest)), 9),
                "maximum_nearest_reference_error_mm": round(float(np.max(nearest)), 6),
            }
            if crosscheck["vertices_within_0_05_mm_fraction"] < 0.98:
                raise ValueError(f"{part_name}: failed assembled-reference cross-check: {crosscheck}")
            cylinder = None
            if "_" in part_name and part_name.rsplit("_", 1)[-1].isdigit():
                cylinder = int(part_name.rsplit("_", 1)[-1])
            manifest["parts"][part_name] = {
                "asset_file": output_path.name,
                "asset_sha256": sha256(output_path),
                "source_group": group_name,
                "source_file": source_path.name,
                "cylinder": cylinder,
                "exploded_to_assembly_translation_mm": correction.tolist(),
                "native_material_name": spec["colour_name"],
                "native_display_colour": spec["display_colour"],
                "vertices": int(len(part_mesh.vertices)),
                "triangles": int(len(part_mesh.faces)),
                "watertight": bool(part_mesh.is_watertight),
                "winding_consistent": bool(part_mesh.is_winding_consistent),
                "bounds_mm": rounded(part_mesh.bounds),
                "extents_mm": rounded(part_mesh.extents),
                "centroid_mm": rounded(part_mesh.centroid),
                "volume_mm3": round(float(abs(part_mesh.volume)), 6),
                "assembled_reference_crosscheck": crosscheck,
            }

    if set(ASSEMBLY_ORDER) != exported_names:
        raise ValueError(
            f"Assembly order and exported assets differ: order={set(ASSEMBLY_ORDER)}, "
            f"assets={exported_names}"
        )
    if len(exported_names) != 12:
        raise ValueError(f"Expected 12 robot-placeable assets, got {len(exported_names)}")

    # Reload every binary STL to validate the serialized deliverables rather
    # than only the in-memory source meshes.
    for part_name, record in manifest["parts"].items():
        reloaded = trimesh.load(output / record["asset_file"], force="mesh", process=True)
        if len(reloaded.faces) != record["triangles"]:
            raise ValueError(f"{part_name}: triangle count changed after STL reload")
        if not reloaded.is_watertight:
            raise ValueError(f"{part_name}: binary STL reload is not watertight")
        if not np.allclose(reloaded.bounds, record["bounds_mm"], atol=1e-5):
            raise ValueError(f"{part_name}: bounds changed after STL reload")

    manifest_path = output / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "asset_count": len(manifest["parts"]),
                "parts": {
                    name: {
                        "triangles": item["triangles"],
                        "bounds_mm": item["bounds_mm"],
                        "watertight": item["watertight"],
                        "reference_match": item["assembled_reference_crosscheck"][
                            "vertices_within_0_05_mm_fraction"
                        ],
                    }
                    for name, item in manifest["parts"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
