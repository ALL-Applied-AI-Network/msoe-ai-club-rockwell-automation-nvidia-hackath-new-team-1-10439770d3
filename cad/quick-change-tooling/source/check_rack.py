"""check_rack.py: checks the Z finger rack (tray_rack.py) against every finger, the gripper and the robot moves.

    python check_rack.py

  1. each rack module is one valid solid;
  2. every Z finger (flat, V and the 20 tips) parks in its slot class: no overlap with the rack, and it really sits on
     the deck (0.05 mm lower would overlap; 0.01 higher is free);
  3. the drop-off / pick-up moves of tray_rack.WAYPOINTS, sampled every 0.5 mm (0.25 in the push), for every finger
     in the served jaw: lower in front, slide in, set down, push to the hard stop (worst case: the program stops
     0.4 short), lift clear, travel along the rack. The receivers, jaws and SMC body, the finger in the other jaw (the
     union of the deepest / widest fingers) and the served finger must not touch the rack or the fingers parked in
     the neighbouring slots (the widest of their class, also across a module joint: N next to W);
  4. standing: each finger's centre of mass (OCC, exact) lies over its wing faces (x 6..12); the margin to the edge
     is reported, below 0.5 mm it is flagged (keep the rack level, or use that tip as a Y finger);
  5. printability, printed on its back (x 30 face on the bed): no downward-facing surface steeper than 45 deg, apart
     from patches at most 4 mm across (the small holes and the 0.5 mm stop lip).
Overlaps are computed with manifold3d on fine meshes (0.02 mm), so a tolerance of 0.001 mm3 applies.
"""
import os, sys
import numpy as np
import trimesh
import manifold3d as mf
sys.path.insert(0, os.environ.get("ATEAM_CAD_HELPERS", os.path.dirname(os.path.abspath(__file__))))
from occ_mesh import to_mesh
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
import check_latch as C
import latch_layout as L
import tray_rack as R
from finger_tips import TIPS
from check_tips import n_solids

TOL = 1e-3
FINGERS = ["flat", "V"] + list(TIPS)
PARTNERS = ["flat", "needle", "flatsd", "wrench10", "rake", "paddle", "socket10", "claw", "hook"]


def man(shape, defl=0.02):
    m = to_mesh(shape, defl); m.merge_vertices()
    M = mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                            tri_verts=np.asarray(m.faces, np.uint32)))
    assert M.status() == mf.Error.NoError and M.volume() > 0, M.status()
    return M


def union(ms):
    out = ms[0]
    for m in ms[1:]:
        out = out + m
    return out


def to_rack(M, x_g, y_s, h_g):
    """gripper frame (finger latched, gripper open) -> rack frame, gripper actuator face at (x_g, y_s, h_g)"""
    T = np.array([[1, 0, 0, x_g], [0, -1, 0, y_s], [0, 0, -1, R.Z_SEAT + h_g - R.H_REST]], float)
    return M.transform(T)


def bbox_hit(a, b):
    A, B = a.bounding_box(), b.bounding_box()
    return all(A[k] < B[k + 3] and B[k] < A[k + 3] for k in range(3))


def ov(a, b):
    return (a ^ b).volume() if bbox_hit(a, b) else 0.0


def shift_z(M, dz):
    return M.translate((0.0, 0.0, dz))


def states():
    """(phase, x_g, dy, h_g, served finger: 'carried' | 'seated' | None)"""
    out = []
    for h in np.arange(R.H_SLIDE + 70.0, R.H_SLIDE - 1e-9, -0.5):
        out.append(("lower in front", -R.SLIDE, 0.0, h, "carried"))
    for x in np.arange(-R.SLIDE, 1e-9, 0.5):
        out.append(("slide in", x, 0.0, R.H_SLIDE, "carried"))
    for h in np.arange(R.H_SLIDE, R.H_REST - 1e-9, -0.2):
        out.append(("set down", 0.0, 0.0, h, "carried"))
    for h in np.arange(R.H_REST, R.H_REST - L.OVERTRAVEL - 1e-9, -0.25):
        out.append(("push", 0.0, 0.0, h, "seated"))
    for h in np.arange(R.H_REST - L.OVERTRAVEL, R.H_CLEAR + 10.0, 0.5):
        out.append(("lift clear", 0.0, 0.0, h, "seated"))
    return out


def main():
    bad = 0
    print("loading meshes ...", flush=True)
    G = {n: man(C.load(n)) for n in ("Receiver_Z", "MHZL2-16D2_jaw", "MHZL2-16D2_body")}
    F = {t: man(C.load(f"Finger_Z_{t}")) for t in FINGERS}
    rot = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]], float)          # the other jaw: 180 deg about Z
    other = union([G["Receiver_Z"].transform(rot), G["MHZL2-16D2_jaw"].transform(rot)]
                  + [F[t].transform(rot) for t in PARTNERS])
    gripper = union([G["Receiver_Z"], G["MHZL2-16D2_jaw"], G["MHZL2-16D2_body"], other])
    cls = {c: [t for t in FINGERS if R.slot_class(t) == c] for c in R.CLASSES}
    park = {c: union([F[t] for t in cls[c]]) for c in cls}          # widest of each class, as a neighbour
    print("slot classes:", {c: cls[c] for c in cls})

    marginal = []
    print("\ncentre of mass over the wing faces (x 6..12, |y| 4..11.8), parked:")
    for t in FINGERS:
        g = GProp_GProps(); BRepGProp.VolumeProperties_s(C.load(f"Finger_Z_{t}"), g); c = g.CentreOfMass()
        x, y, h = c.X(), -c.Y(), R.Z_SEAT - c.Z()
        m = min(x - 6.0, 12.0 - x)
        flag = "" if m >= 0.5 else ("  MARGINAL" if m > 0 else "  FALLS OVER")
        marginal += [t] if flag else []
        print(f"  {t:9s} CG x {x:6.2f} y {y:5.2f} h {h:5.2f}: {m:4.2f} mm inside the support edge{flag}")
        bad += m <= 0
    for name, classes in R.MODULES.items():
        sh = C.load(f"Rack_Z_{name}")
        ok = BRepCheck_Analyzer(sh).IsValid() and n_solids(sh) == 1
        rack = man(sh)
        ys, W = R.layout(classes)
        print(f"\nRack_Z_{name}: valid single solid {ok}, volume {rack.volume():.0f} mm3, {len(ys)} slots")
        bad += not ok
        # 2. parking
        c0 = classes[0]
        for t in cls[c0]:
            fin = to_rack(F[t], 0.0, ys[0], R.H_REST)
            o = ov(fin, rack); lo = ov(shift_z(fin, -0.05), rack); hi = ov(shift_z(fin, 0.01), rack)
            okp = o < TOL and hi < TOL and lo > TOL
            bad += not okp
            print(f"  park {t:9s}: overlap {o:.4f}, 0.01 up {hi:.4f}, 0.05 down {lo:.3f} mm3 "
                  + ("OK (seated)" if okp else "FAIL"), flush=True)
        # 3. moves, served slot = an inner one and the end one (module joint to the other class)
        other_c = "W" if c0 == "N" else "N"
        pitch_same = 2 * R.CLASSES[c0]["half"]
        pitch_mix = R.CLASSES[c0]["half"] + R.CLASSES[other_c]["half"]
        cases = [(f"inner slot (neighbours {c0} {c0})", 1, [(-pitch_same, c0), (pitch_same, c0)]),
                 (f"end slot (neighbours {c0}, {other_c} across the joint)", len(ys) - 1,
                  [(-pitch_same, c0), (pitch_mix, other_c)])]
        for label, k, neigh in cases:
            ysl = ys[k]
            stat = union([rack] + [to_rack(park[c], 0.0, ysl + dy, R.H_REST) for dy, c in neigh])
            worst = {}; fails = []
            for ph, x_g, dy, h_g, sv in states():
                gp = to_rack(gripper, x_g, ysl + dy, h_g)
                o = ov(gp, stat)
                if o > TOL:
                    fails.append((ph, "gripper / other finger", round(x_g, 2), round(h_g, 2), round(o, 3)))
                for t in cls[c0]:
                    if sv == "carried":
                        o = ov(to_rack(F[t], x_g, ysl + dy, h_g), stat)
                    else:
                        o = 0.0
                    worst[t] = max(worst.get(t, 0.0), o)
                    if o > TOL:
                        fails.append((ph, t, round(x_g, 2), round(h_g, 2), round(o, 3)))
            # seated served finger vs parked neighbours, and travel along the rack at the clear height
            for t in cls[c0]:
                o = ov(to_rack(F[t], 0.0, ysl, R.H_REST), stat)
                if o > TOL:
                    fails.append(("parked", t, 0, R.H_REST, round(o, 3)))
            for dy in np.arange(-pitch_same, pitch_mix + 1e-9, 0.5):
                o = ov(to_rack(gripper, 0.0, ysl + dy, R.H_CLEAR), stat)
                if o > TOL:
                    fails.append(("travel", "gripper / other finger", 0, R.H_CLEAR, round(o, 3)))
                    break
            bad += bool(fails)
            print(f"  moves, {label}: {len(states())} gripper poses x {len(cls[c0])} fingers -> "
                  + ("OK" if not fails else f"{len(fails)} FAIL, e.g. {fails[:4]}"), flush=True)
        # 4. printability
        m = to_mesh(sh, 0.02)
        n, c_, a = m.face_normals, m.triangles_center, m.area_faces
        down = (n[:, 0] > 0.72) & (c_[:, 0] < R.X_BACK1 - 0.01)
        big = small = 0.0
        if down.any():
            sub = m.submesh([np.where(down)[0]], append=True)
            for part in sub.split(only_watertight=False):
                ext = part.bounds[1] - part.bounds[0]
                if min(ext[1], ext[2]) <= 4.0:
                    small += part.area
                else:
                    big += part.area
        okp = big < 1.0
        bad += not okp
        print(f"  print on its back: overhangs needing supports {big:.1f} mm2, small ledges / holes {small:.1f} mm2 "
              + ("OK" if okp else "FAIL"))
    if marginal:
        print(f"\nstand on a margin under 0.5 mm (rack must be level; better used as Y fingers): {', '.join(marginal)}")
    print("\nRACK OK" if not bad else f"\n{bad} RACK CHECKS FAILED")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
