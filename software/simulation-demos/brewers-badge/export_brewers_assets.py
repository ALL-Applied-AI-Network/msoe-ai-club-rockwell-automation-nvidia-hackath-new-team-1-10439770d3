#!/usr/bin/env python3
"""Export the five Brewers assembly components from 3MF to exact binary STL.

The 3MF inputs are kept as the source of truth.  This converter applies any
3MF scene transforms, merges scene geometry without changing units, emits one
watertight STL per robot-placeable component, and records auditable geometry
and hash metadata for the Isaac Sim run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import trimesh


PARTS = (
    ("stand", "Stand.3mf", 0.0),
    ("base", "Navybase.3mf", 3.0),
    ("emblem", "Gloveemblem.3mf", 4.3),
    ("detail", "Rim and lettering.3mf", 4.3),
    ("nameplate", "Nameplate.3mf", 3.8),
)

CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
NS = {"m": CORE_NS}
KNOWN_COLOURS = {
    "13294B": "navy",
    "FFC52F": "gold",
    "FFFFFF": "white",
    "4169E1": "royalblue",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_mesh(loaded: Any) -> trimesh.Trimesh:
    """Return a single mesh with scene transforms applied."""
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()
    if not isinstance(loaded, trimesh.Scene):
        raise TypeError(f"Unsupported trimesh load result: {type(loaded)!r}")
    meshes = []
    for node_name in loaded.graph.nodes_geometry:
        transform, geometry_name = loaded.graph[node_name]
        mesh = loaded.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    if not meshes:
        raise ValueError("3MF scene contained no mesh geometry")
    return trimesh.util.concatenate(meshes)


def rounded(values: np.ndarray, digits: int = 6) -> list:
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def colour_name(display_colour: str, material_index: int) -> str:
    rgb = display_colour.lstrip("#")[:6].upper()
    return KNOWN_COLOURS.get(rgb, f"material_{material_index}_{rgb.lower()}")


def load_3mf_colour_regions(path: Path) -> tuple[trimesh.Trimesh, list[dict[str, Any]]]:
    """Read the native 3MF triangle material IDs without losing face order.

    Generic mesh loaders correctly recover geometry but do not expose the
    OpenSCAD 3MF ``p1`` per-triangle base-material assignment.  Reading the
    compact XML directly preserves that assignment exactly.  These isolated
    exports contain one mesh object and no build transform; explicit checks
    keep the assumption auditable rather than silently changing geometry.
    """
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("3D/3dmodel.model"))
    if root.attrib.get("unit", "millimeter") != "millimeter":
        raise ValueError(f"{path.name}: expected millimetre 3MF units")

    objects = [obj for obj in root.findall(".//m:object", NS) if obj.find("m:mesh", NS) is not None]
    if len(objects) != 1:
        raise ValueError(f"{path.name}: expected one mesh object, found {len(objects)}")
    obj = objects[0]
    items = root.findall(".//m:build/m:item", NS)
    if len(items) != 1 or items[0].attrib.get("objectid") != obj.attrib.get("id"):
        raise ValueError(f"{path.name}: build does not reference its single mesh object")
    if "transform" in items[0].attrib:
        raise ValueError(f"{path.name}: unexpected build transform")

    vertices = np.asarray([
        [float(v.attrib[axis]) for axis in ("x", "y", "z")]
        for v in obj.findall("m:mesh/m:vertices/m:vertex", NS)
    ], dtype=np.float64)
    triangle_elements = obj.findall("m:mesh/m:triangles/m:triangle", NS)
    faces = np.asarray([
        [int(t.attrib[key]) for key in ("v1", "v2", "v3")]
        for t in triangle_elements
    ], dtype=np.int64)
    whole = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    material_sets: dict[str, list[dict[str, str]]] = {}
    for materials in root.findall(".//m:basematerials", NS):
        material_sets[materials.attrib["id"]] = [dict(base.attrib) for base in materials]
    pid = obj.attrib.get("pid")
    if pid not in material_sets:
        raise ValueError(f"{path.name}: object has no usable base-material set")
    bases = material_sets[pid]
    default_index = int(obj.attrib.get("pindex", "0"))

    by_index: dict[int, list[int]] = {}
    for face_index, triangle in enumerate(triangle_elements):
        # OpenSCAD assigns one material to the full triangle using p1.  Assert
        # that no per-vertex override would be discarded.
        p1 = int(triangle.attrib.get("p1", default_index))
        if int(triangle.attrib.get("p2", p1)) != p1 or int(triangle.attrib.get("p3", p1)) != p1:
            raise ValueError(f"{path.name}: per-vertex material interpolation is unsupported")
        by_index.setdefault(p1, []).append(face_index)

    regions = []
    for material_index in sorted(by_index):
        if material_index >= len(bases):
            raise ValueError(f"{path.name}: material index {material_index} is out of range")
        base = bases[material_index]
        display_colour = base.get("displaycolor", "#000000")
        region_mesh = trimesh.Trimesh(
            vertices=vertices.copy(),
            faces=faces[np.asarray(by_index[material_index], dtype=np.int64)].copy(),
            process=False,
        )
        region_mesh.remove_unreferenced_vertices()
        regions.append({
            "name": colour_name(display_colour, material_index),
            "material_index": material_index,
            "material_label": base.get("name", f"material {material_index}"),
            "display_colour": display_colour,
            "mesh": region_mesh,
        })
    return whole, regions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "cad" / "brewers-badge",
        help="Folder containing the five isolated 3MF exports",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "assets",
        help="Destination for STL assets and manifest",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema": "ateam.brewers_badge_assets.v1",
        "units": "millimetres",
        "source_directory": str(source),
        "assembly_order": [name for name, _, _ in PARTS],
        "assembly_offsets_mm": {
            name: [0.0, 0.0, z_mm] for name, _, z_mm in PARTS
        },
        "parts": {},
    }

    for name, source_name, z_mm in PARTS:
        source_path = source / source_name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        mesh = as_mesh(trimesh.load(source_path, force=None, process=False))
        xml_mesh, colour_regions = load_3mf_colour_regions(source_path)
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError(f"{source_name} produced an empty mesh")
        if len(xml_mesh.faces) != len(mesh.faces) or not np.allclose(xml_mesh.bounds, mesh.bounds, atol=1e-9):
            raise ValueError(f"{source_name}: direct 3MF geometry disagrees with trimesh load")

        # Avoid topology-altering repair/decimation.  Removing unreferenced
        # vertices is representation-only and makes the audit deterministic.
        mesh.remove_unreferenced_vertices()
        output_path = output / f"{name}.stl"
        output_path.write_bytes(trimesh.exchange.stl.export_stl(mesh))

        components = mesh.split(only_watertight=False)
        region_records: dict[str, Any] = {}
        for region in colour_regions:
            region_name = region["name"]
            if region_name in region_records:
                raise ValueError(f"{source_name}: duplicate friendly colour name {region_name}")
            region_mesh = region.pop("mesh")
            region_path = output / f"{name}_{region_name}.stl"
            region_path.write_bytes(trimesh.exchange.stl.export_stl(region_mesh))
            region_components = region_mesh.split(only_watertight=False)
            region_records[region_name] = {
                **region,
                "asset_file": region_path.name,
                "asset_sha256": sha256(region_path),
                "vertices": int(len(region_mesh.vertices)),
                "triangles": int(len(region_mesh.faces)),
                "connected_components": int(len(region_components)),
                "watertight": bool(region_mesh.is_watertight),
                "bounds_mm": rounded(region_mesh.bounds),
                "extents_mm": rounded(region_mesh.extents),
            }
        if sum(region["triangles"] for region in region_records.values()) != len(mesh.faces):
            raise ValueError(f"{source_name}: colour-region triangle counts do not cover the source")

        manifest["parts"][name] = {
            "source_file": source_name,
            "source_sha256": sha256(source_path),
            "asset_file": output_path.name,
            "asset_sha256": sha256(output_path),
            "assembly_offset_mm": [0.0, 0.0, z_mm],
            "vertices": int(len(mesh.vertices)),
            "triangles": int(len(mesh.faces)),
            "connected_components": int(len(components)),
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "bounds_mm": rounded(mesh.bounds),
            "extents_mm": rounded(mesh.extents),
            "centroid_mm": rounded(mesh.centroid),
            "volume_mm3": round(float(abs(mesh.volume)), 6),
            "colour_regions": region_records,
        }

    manifest_path = output / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest_path),
        "parts": {
            name: {
                "extents_mm": data["extents_mm"],
                "watertight": data["watertight"],
                "connected_components": data["connected_components"],
                "colour_region_triangles": {
                    colour: region["triangles"]
                    for colour, region in data["colour_regions"].items()
                },
            }
            for name, data in manifest["parts"].items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
