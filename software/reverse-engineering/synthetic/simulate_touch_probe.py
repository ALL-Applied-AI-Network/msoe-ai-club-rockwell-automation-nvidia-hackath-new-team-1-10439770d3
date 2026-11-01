"""Capture-side idealized contact probe observations from evaluator geometry.

This program belongs to sensor generation, not reconstruction. It reads the
withheld source once, emits noisy axis-extreme contacts, and reconstruction is
then restricted to the resulting observation JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--noise-mm", type=float, default=0.02)
parser.add_argument("--seed", type=int, default=20261024)
args = parser.parse_args()

root = Path(args.dataset).resolve()
source = np.load(root / "evaluator_only" / "source_mesh.npz")
vertices = np.asarray(source["vertices"], dtype=np.float64)
rng = np.random.default_rng(args.seed)
lo, hi = vertices.min(axis=0), vertices.max(axis=0)
contacts = []
measured_lo, measured_hi = lo.copy(), hi.copy()
sigma = args.noise_mm / 1000.0
for axis in range(3):
    measured_lo[axis] += rng.normal(0.0, sigma)
    measured_hi[axis] += rng.normal(0.0, sigma)
    for side, value, direction in (("min", measured_lo[axis], +1.0), ("max", measured_hi[axis], -1.0)):
        point = np.median(vertices, axis=0)
        point[axis] = value
        ray = np.zeros(3)
        ray[axis] = direction
        contacts.append({"axis": axis, "side": side, "point_m": point.tolist(), "approach_direction": ray.tolist()})

payload = {
    "sensor": "idealized guarded contact probe",
    "noise_sigma_mm": args.noise_mm,
    "bounds_m": [measured_lo.tolist(), measured_hi.tolist()],
    "contacts": contacts,
    "ground_truth_mesh_exposed_to_reconstructor": False,
    "caveat": "axis-extreme contacts only; internal bores and mating planes are not probed",
}
(root / "observations" / "touch_probe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
