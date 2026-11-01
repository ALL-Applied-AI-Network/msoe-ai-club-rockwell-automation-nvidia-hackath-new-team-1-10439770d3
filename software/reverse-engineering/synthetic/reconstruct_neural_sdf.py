"""Kinematic-seeded compact neural SDF reconstruction.

This is an independently implemented, observation-only neural implicit
baseline. It is not presented as NeuS2/Neuralangelo: it uses reported FANUC
poses and noisy RGB-D surface samples, Fourier features, bounded pose priors,
normal-offset SDF supervision, and an Eikonal regularizer. Its purpose is to
test the neural-implicit family without silently substituting ground-truth
mesh supervision or claiming an unavailable upstream stack was executed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from skimage import measure
import torch
from torch import nn
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruct_classical import fused_points, intrinsics, load_manifest, select_frames  # noqa: E402


class FourierSDF(nn.Module):
    def __init__(self, levels: int = 8, width: int = 160):
        super().__init__()
        self.levels = levels
        input_size = 3 * (1 + 2 * levels)
        self.net = nn.Sequential(
            nn.Linear(input_size, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width // 2), nn.SiLU(),
            nn.Linear(width // 2, 1),
        )

    def encode(self, xyz: torch.Tensor) -> torch.Tensor:
        bands = 2.0 ** torch.arange(self.levels, device=xyz.device, dtype=xyz.dtype) * math.pi
        phase = xyz[..., None] * bands
        return torch.cat((xyz, torch.sin(phase).flatten(-2), torch.cos(phase).flatten(-2)), dim=-1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.net(self.encode(xyz)).squeeze(-1)


def export_all(mesh: trimesh.Trimesh, stem: Path) -> dict:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix in ("ply", "obj", "stl"):
        path = stem.with_suffix(f".{suffix}")
        mesh.export(path)
        outputs[suffix] = path.name
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=1100)
    parser.add_argument("--batch", type=int, default=32768)
    parser.add_argument("--surface-voxel-mm", type=float, default=0.70)
    parser.add_argument("--grid-mm", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root = Path(args.dataset).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    manifest = load_manifest(root)
    intrinsic, _ = intrinsics(manifest)
    frames = select_frames(manifest["frames"], 0, include_flipped=False)
    cloud = fused_points(root / "observations", frames, intrinsic, args.surface_voxel_mm / 1000.0)
    points = np.asarray(cloud.points, dtype=np.float32)
    normals = np.asarray(cloud.normals, dtype=np.float32)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1) & (np.linalg.norm(normals, axis=1) > 0.5)
    points, normals = points[valid], normals[valid]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    if len(points) < 2000:
        raise RuntimeError(f"insufficient oriented surface samples: {len(points)}")

    lo = np.quantile(points, 0.001, axis=0) - 0.006
    hi = np.quantile(points, 0.999, axis=0) + 0.006
    centre = (lo + hi) * 0.5
    scale = float(np.max(hi - lo) * 0.55)
    p_norm = (points - centre) / scale
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ``np.quantile`` promotes ``lo``/``hi`` (and therefore ``p_norm``) to
    # float64.  Keep the observation coordinates aligned with the network's
    # float32 weights; otherwise PyTorch fails on the first Linear layer with
    # a Double-vs-Float matrix multiplication error.
    p_tensor = torch.from_numpy(p_norm).float().to(device)
    n_tensor = torch.from_numpy(normals).float().to(device)
    model = FourierSDF().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1e-6)
    losses = []
    for step in range(args.steps):
        index = torch.randint(0, len(p_tensor), (args.batch,), device=device)
        surface = p_tensor[index]
        normal = n_tensor[index]
        # Offset is expressed in normalized coordinates; positive follows the
        # outward (toward-camera) normal and negative moves inside.
        offset_m = (torch.rand(args.batch, device=device) * 2.0 - 1.0) * 0.0045
        offset = offset_m / scale
        query = surface + normal * offset[:, None]
        prediction = model(query)
        loss_sdf = torch.nn.functional.smooth_l1_loss(prediction, offset, beta=0.002 / scale)
        surface_prediction = model(surface[: args.batch // 4])
        loss_surface = surface_prediction.abs().mean()
        loss = loss_sdf + 0.75 * loss_surface
        if step % 4 == 0:
            eikonal = (torch.rand((4096, 3), device=device) * 2.0 - 1.0).requires_grad_(True)
            value = model(eikonal)
            gradient = torch.autograd.grad(value.sum(), eikonal, create_graph=True)[0]
            loss_eikonal = ((gradient.norm(dim=1) - 1.0) ** 2).mean()
            loss = loss + 0.04 * loss_eikonal
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step + 1 == args.steps:
            losses.append({"step": step, "loss": float(loss.detach().cpu())})
            print(f"[neural-sdf] {step}/{args.steps} loss={losses[-1]['loss']:.6f}", flush=True)

    spacing = args.grid_mm / 1000.0
    dims = np.maximum(3, np.ceil((hi - lo) / spacing).astype(int) + 1)
    xs = np.linspace(lo[0], hi[0], dims[0], dtype=np.float32)
    ys = np.linspace(lo[1], hi[1], dims[1], dtype=np.float32)
    zs = np.linspace(lo[2], hi[2], dims[2], dtype=np.float32)
    field = np.empty(tuple(dims), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for x0 in range(0, dims[0], 8):
            x1 = min(dims[0], x0 + 8)
            gx, gy, gz = np.meshgrid(xs[x0:x1], ys, zs, indexing="ij")
            xyz = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
            tensor = torch.from_numpy((xyz - centre) / scale).float().to(device)
            values = []
            for start_index in range(0, len(tensor), 262144):
                values.append(model(tensor[start_index:start_index + 262144]).cpu().numpy())
            field[x0:x1] = np.concatenate(values).reshape((x1 - x0, dims[1], dims[2]))
    if not (float(field.min()) <= 0.0 <= float(field.max())):
        raise RuntimeError(f"neural field has no zero crossing: [{field.min()}, {field.max()}]")
    vertices, faces, normals_mc, _ = measure.marching_cubes(
        field, level=0.0,
        spacing=((hi[0] - lo[0]) / (dims[0] - 1),
                 (hi[1] - lo[1]) / (dims[1] - 1),
                 (hi[2] - lo[2]) / (dims[2] - 1)),
    )
    vertices += lo
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals_mc, process=True)
    outputs = export_all(mesh, out / "neural_sdf_depth")
    report = {
        "method": "neural_sdf_depth",
        "status": "completed",
        "runtime_s": time.perf_counter() - start,
        "device": str(device),
        "input_frames": len(frames),
        "input_oriented_points": int(len(points)),
        "training_steps": args.steps,
        "grid_dimensions": dims.tolist(),
        "grid_spacing_mm": args.grid_mm,
        "loss_trace": losses,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "components": int(len(mesh.split(only_watertight=False))),
        "outputs": outputs,
        "truth_mesh_access": False,
        "identity": "compact Fourier-feature neural SDF; not NeuS2/Neuralangelo",
    }
    (out / "neural_sdf_depth.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
