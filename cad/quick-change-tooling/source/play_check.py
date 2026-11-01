"""play_check.py: how far a latched finger can move inside its receiver because of the sliding-fit clearances, and how
much that moves the fingertip.

    python play_check.py Y|Z

Rigid moves of the finger (receiver fixed), each found by bisection to the first contact with the receiver
(overlap > 1e-5 mm3):
  - translations along the two axes across the insertion direction;
  - rotations about lines parallel to Y (tip moves in X, the grip direction) and parallel to X (tip moves in Y), through
    a grid of pivot points on the joint: a free finger rocks about whichever contact lets it go furthest.
Tip displacement = movement of the gripping-face centre at the fingertip (X 6, Y 0, Z 128.7).
"""
import sys, math
from OCP.gp import gp_Trsf, gp_Ax1, gp_Pnt, gp_Dir, gp_Vec
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
import check_latch as C
import latch_layout as L

TIP = (L.FACE_OPEN, 0.0, L.TIP_END)
EPS_V = 1e-5


def moved(shape, T):
    return BRepBuilderAPI_Transform(shape, T, True).Shape()


def rot(axis_pt, axis_dir, ang):
    T = gp_Trsf(); T.SetRotation(gp_Ax1(gp_Pnt(*axis_pt), gp_Dir(*axis_dir)), ang); return T


def trans(v):
    T = gp_Trsf(); T.SetTranslation(gp_Vec(*v)); return T


def apply(T, p):
    q = gp_Pnt(*p).Transformed(T); return (q.X(), q.Y(), q.Z())


def first_contact(fin, rec, make_T, hi, n_scan=60, iters=18):
    """largest parameter t in [0, hi] before the first contact; make_T(t) -> gp_Trsf.
    Scans forward from 0 and bisects only inside the first bracket that shows contact: OpenCascade's Boolean can
    return an empty result for deep, many-face intersections (seen at 2 mm), so a far endpoint is never trusted."""
    lo, hi_found = 0.0, None
    for k in range(1, n_scan + 1):
        t = hi * k / n_scan
        if C.overlap(moved(fin, make_T(t)), rec) > EPS_V:
            hi_found = t; break
        lo = t
    if hi_found is None:
        return hi, False
    hi = hi_found
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if C.overlap(moved(fin, make_T(mid)), rec) > EPS_V:
            hi = mid
        else:
            lo = mid
    return lo, True


def main(var):
    v = L.VARIANTS[var]
    rec, fin = C.load(f"Receiver_{var}"), C.load(f"Finger_{var}_flat")
    print(f"=== variant {var}: latched finger, receiver fixed (tip point {TIP})")
    assert C.overlap(fin, rec) <= EPS_V
    axes = [a for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1)) if abs(sum(x * y for x, y in zip(a, v.ins))) < 0.5]
    res = {}
    for a in axes:
        for sgn in (1, -1):
            d = tuple(sgn * x for x in a)
            t, hit = first_contact(fin, rec, lambda s: trans(tuple(s * x for x in d)), 1.2)
            name = "+-"[sgn < 0] + "XYZ"[a.index(1)]
            res[f"slide {name}"] = (t, t * abs(d[0]), t * abs(d[1]))
            print(f"  slide {name}: {t:.3f} mm{'' if hit else ' (no contact within 1.2 mm!)'}")
    # joint extent of the finger (where it overlaps the receiver's Z range)
    bf, br = C.box(fin), C.box(rec)
    z0, z1 = max(bf[2], br[2]), min(bf[5], br[5])
    zs = [z0, 0.5 * (z0 + z1), z1]
    xs = [L.FINGER_IN, L.X_I, L.X_TF, L.BEAM_X1]
    ys = [bf[1], 0.0, bf[4]]
    worst_x = worst_y = (0.0, None)
    for sgn in (1, -1):
        for x in xs:
            for z in zs:
                th, hit = first_contact(fin, rec, lambda t: rot((x, 0, z), (0, sgn, 0), t), 0.06)
                dx = abs(apply(rot((x, 0, z), (0, sgn, 0), th), TIP)[0] - TIP[0])
                if dx > worst_x[0]:
                    worst_x = (dx, f"about Y through X {x:.2f}, Z {z:.2f}, {'+-'[sgn < 0]}: {math.degrees(th):.3f} deg")
        for y in ys:
            for z in zs:
                th, hit = first_contact(fin, rec, lambda t: rot((0, y, z), (sgn, 0, 0), t), 0.06)
                dy = abs(apply(rot((0, y, z), (sgn, 0, 0), th), TIP)[1] - TIP[1])
                if dy > worst_y[0]:
                    worst_y = (dy, f"about X through Y {y:.2f}, Z {z:.2f}, {'+-'[sgn < 0]}: {math.degrees(th):.3f} deg")
    sx = max(r[1] for k, r in res.items()); sy = max(r[2] for k, r in res.items())
    print(f"  worst rocking, tip moves in X (grip direction): {worst_x[0]:.3f} mm  ({worst_x[1]})")
    print(f"  worst rocking, tip moves in Y (across):          {worst_y[0]:.3f} mm  ({worst_y[1]})")
    print(f"  => fingertip play, grip direction X: up to {worst_x[0]:.2f} mm by rocking, {sx:.2f} mm by sliding; "
          f"across Y: up to {worst_y[0]:.2f} mm rocking, {sy:.2f} mm sliding")
    return worst_x[0], worst_y[0]


if __name__ == "__main__":
    for var in sys.argv[1:] or ["Y", "Z"]:
        main(var)
