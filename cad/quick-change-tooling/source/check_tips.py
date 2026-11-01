"""check_tips.py: checks every tip finger (finger_tips.py) on its SolidWorks STEP export.

    python check_tips.py [Y|Z] [tip ...]        (default: both variants, every tip)

Per finger:
  1. one valid solid; nothing below X 6 (the gripping face = print bed);
  2. the latch end is the proven one: the finger below Z 119.5 has exactly the volume of Finger_<v>_flat there, and
     their difference there is empty;
  3. the tip clears the receiver, jaw and SMC body on the way in, latched, at the hard stop and 0.8 mm past it;
  4. with its partner on the other jaw (finger_tips.TIPS "pair" column): no collision at full open or full closed;
  5. print orientation (gripping face down, X up): downward-facing surface steeper than 45 deg in the tip, which would
     need supports. Flat bridges up to 10 mm span print fine and are reported separately.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get("ATEAM_CAD_HELPERS", os.path.dirname(os.path.abspath(__file__))))
from occ_mesh import to_mesh
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
import check_latch as C
import latch_layout as L
from finger_tips import TIPS, TE

Z_TIP = 119.5
TOL = 1e-3


def bigbox(z0, z1):
    return BRepPrimAPI_MakeBox(gp_Pnt(-200, -200, z0), gp_Pnt(200, 200, z1)).Shape()


def n_solids(s):
    e = TopExp_Explorer(s, TopAbs_SOLID); n = 0
    while e.More():
        n += 1; e.Next()
    return n


def common(a, b):
    from OCP.TopTools import TopTools_ListOfShape
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    la, lb = TopTools_ListOfShape(), TopTools_ListOfShape(); la.Append(a); lb.Append(b)
    op = BRepAlgoAPI_Common(); op.SetArguments(la); op.SetTools(lb); op.SetNonDestructive(True); op.Build()
    return op.Shape()


def cut(a, b):
    op = BRepAlgoAPI_Cut(a, b); op.Build(); return op.Shape()


def overhangs(tip_shape):
    """area (mm2) of downward-facing surface steeper than 45 deg, split into big (needs supports) and small bridges"""
    m = to_mesh(tip_shape, 0.02)
    n = m.face_normals; c = m.triangles_center; a = m.area_faces
    down = (n[:, 0] < -0.72) & (c[:, 0] > 6.05)
    if not down.any():
        return 0.0, 0.0, 0.0
    # group the downward faces into connected patches; a patch at most 10 mm across is a bridge (PETG bridges that easily)
    import trimesh
    sub = m.submesh([np.where(down)[0]], append=True)
    big = small = 0.0; worst = 0.0
    for part in sub.split(only_watertight=False):
        ext = part.bounds[1] - part.bounds[0]
        span = min(ext[1], ext[2])                  # in the bed plane (Y, Z)
        if span <= 10.0:
            small += part.area
        else:
            big += part.area; worst = max(worst, span)
    return big, small, worst


def main(variants, tips):
    v_ = {"Y": L.VarY, "Z": L.VarZ}
    body, jaw = C.load("MHZL2-16D2_body"), C.load("MHZL2-16D2_jaw")
    total_bad = 0; rows = []
    for vn in variants:
        v = v_[vn]
        rec = C.load(f"Receiver_{vn}"); flat = C.load(f"Finger_{vn}_flat")
        root_ref = C.overlap(flat, bigbox(-1000, Z_TIP))
        for t in tips:
            name = f"Finger_{vn}_{t}"
            fin = C.load(name); bad = []
            ok_solid = BRepCheck_Analyzer(fin).IsValid() and n_solids(fin) == 1
            if not ok_solid:
                bad.append("not one valid solid")
            b = C.box(fin)
            if b[0] < 6.0 - 1e-4:
                bad.append(f"material below X 6 (min X {b[0]:.3f})")
            # 2. root identical
            root = C.overlap(fin, bigbox(-1000, Z_TIP))
            diff = C.V(cut(common(fin, bigbox(-1000, Z_TIP)), flat)) if abs(root - root_ref) < 1.0 else 999
            if abs(root - root_ref) > TOL or diff > TOL:
                bad.append(f"latch end differs (vol {root:.3f} vs {root_ref:.3f}, extra {diff:.4f})")
            tip = common(fin, bigbox(Z_TIP, 1000))
            # 3. clearance through the stroke
            worst = 0.0
            for s in (-25.0, -10.0, 0.0, L.OVERTRAVEL, L.OVERTRAVEL + 0.8):
                T = C.finger_at(tip, v, s)
                for other in (rec, jaw, body):
                    worst = max(worst, C.overlap(T, other))
            if worst > TOL:
                bad.append(f"tip hits receiver/jaw/body in the stroke ({worst:.3f} mm3)")
            # 4. partner on the other jaw
            partner_t = TIPS[t][2].split()[0]
            partner = fin if partner_t == t else C.load(f"Finger_{vn}_{partner_t}")
            pr, pj = C.rotz180(partner), C.rotz180(jaw)
            prec = C.rotz180(rec)
            gaps = []
            for closing in (0.0, 6.0):
                A = C.tr(fin, (-closing, 0, 0))
                B_ = [C.tr(x, (closing, 0, 0)) for x in (pr, pj, prec)]
                o = max(C.overlap(A, x) for x in B_)
                o2 = max(C.overlap(C.tr(partner, (-closing, 0, 0)), C.tr(x, (closing, 0, 0)))
                         for x in (C.rotz180(fin),)) if partner_t != t else 0.0
                if max(o, o2) > TOL:
                    bad.append(f"collides with partner '{partner_t}' at closing {closing} ({max(o, o2):.3f} mm3)")
                gaps.append(C.gap(A, B_[0]))
            # 5. print
            big, small, span = overhangs(tip)
            if big > 1.0:
                bad.append(f"overhang needing supports: {big:.1f} mm2 (span {span:.1f} mm)")
            out = b[5] - TE
            rows.append((name, out, gaps[1], small, bad))
            total_bad += bool(bad)
            print(f"{name:24s} sticks out {out:5.1f} mm past Z 128.7 | partner {partner_t:8s} gap closed {gaps[1]:.3f} | "
                  f"bridges {small:5.1f} mm2 | " + ("OK" if not bad else "FAIL: " + "; ".join(bad)), flush=True)
    print("ALL TIPS OK" if not total_bad else f"{total_bad} FINGERS FAILED")
    return total_bad


if __name__ == "__main__":
    args = sys.argv[1:]
    vs = [a for a in args if a in ("Y", "Z")] or ["Y", "Z"]
    ts = [a for a in args if a in TIPS] or list(TIPS)
    sys.exit(1 if main(vs, ts) else 0)
