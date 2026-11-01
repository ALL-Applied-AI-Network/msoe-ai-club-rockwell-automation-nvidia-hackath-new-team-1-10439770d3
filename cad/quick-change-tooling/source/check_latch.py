"""check_latch.py: functional checks of the SolidWorks parts (their own STEP exports), OpenCascade, exact solids.

    python check_latch.py Y|Z

1. Latch cycle: the finger is moved along the simulated pin path (simulate.py: insert, latch, push, release) with the
   shuttle following the pin's lateral position. At every sample: finger x shuttle, finger x receiver and
   shuttle x receiver must not overlap.
2. Retention: latched, the finger pulled 0.3 mm outward (or the shuttle pushed 0.3 mm sideways) MUST collide with the
   peg: the latch really holds.
3. Gripper: SMC body + both jaws + both latched fingers, receivers and shuttles, at full open, holding a 10 mm cube and
   full closed: no collisions (SMC's own jaw/body contacts are by design and skipped).
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from OCP.gp import gp_Trsf, gp_Vec, gp_Ax1, gp_Pnt, gp_Dir
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from analyze_step import solids
import latch_layout as L
import simulate as S
import track_geom as TG

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
TOL = 1e-4                                          # mm3


def V(s):
    p = GProp_GProps(); BRepGProp.VolumeProperties_s(s, p); return p.Mass()


def load(name):
    return solids(os.path.join(OUT, name + ".STEP"))[0]


def tr(shape, t=(0, 0, 0), rot=None):
    """rot: 3x3 rows (component axes in world) applied first, then translation t"""
    T = gp_Trsf()
    if rot is not None:
        (a, b, c), (d, e, f), (g_, h, i) = rot
        T.SetValues(a, d, g_, t[0], b, e, h, t[1], c, f, i, t[2])   # columns = images of the local axes
    else:
        T.SetTranslation(gp_Vec(*t))
    return BRepBuilderAPI_Transform(shape, T, True).Shape()


def rotz180(shape):
    T = gp_Trsf(); T.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), math.pi)
    return BRepBuilderAPI_Transform(shape, T, True).Shape()


def box(s):
    b = Bnd_Box(); BRepBndLib.Add_s(s, b); return b.Get()


def disjoint_boxes(a, b):
    A, B = box(a), box(b)
    return any(A[k + 3] < B[k] - 1e-6 or B[k + 3] < A[k] - 1e-6 for k in range(3))


def overlap(a, b):
    """intersection volume. Non-destructive: by default OpenCascade's Boolean may enlarge the tolerances of its INPUT
    shapes in place, so after hundreds of calls on the same solids the answers drift (seen: a false 'no contact')."""
    if disjoint_boxes(a, b):
        return 0.0
    from OCP.TopTools import TopTools_ListOfShape
    la, lb = TopTools_ListOfShape(), TopTools_ListOfShape()
    la.Append(a); lb.Append(b)
    op = BRepAlgoAPI_Common()
    op.SetArguments(la); op.SetTools(lb); op.SetNonDestructive(True); op.SetRunParallel(False)
    op.Build()
    if not op.IsDone():
        raise RuntimeError("Boolean common failed")
    return V(op.Shape())


def gap(a, b):
    return BRepExtrema_DistShapeShape(a, b).Value()


def rot_matrix(name):
    """shuttle local -> world (world = M @ local). Y variant: identity (local Z = u = world Z).
    Z variant: local Z (u) -> world +Y, local Y (d) -> world -Z. The shuttle is symmetric, so the signs don't matter."""
    if name == "Y":
        return ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    return ((1, 0, 0), (0, 0, 1), (0, -1, 0))


def tr_m(shape, M, t):
    T = gp_Trsf()
    T.SetValues(M[0][0], M[0][1], M[0][2], t[0], M[1][0], M[1][1], M[1][2], t[1], M[2][0], M[2][1], M[2][2], t[2])
    return BRepBuilderAPI_Transform(shape, T, True).Shape()


def shuttle_at(sh, v, u):
    x, y, z = v.shuttle_at(u)
    return tr_m(sh, rot_matrix(v.name), (0.0, y, z))


def finger_at(fn, v, s):
    return tr(fn, tuple(s * c for c in v.ins))


def cycle_samples(step=0.25):
    D = TG.derived()
    sim = S.Sim(S.free_for(0.0))
    out = []
    for u0, tag in ((D["u1"] - 0.3, "R"), (D["u0"] + 0.3, "L")):
        ok, res, paths = S.cycle(sim, u0, 0.4, D)
        assert ok, res
        for ph, path in enumerate(paths):
            last = None
            for p in path:
                if last is None or math.hypot(p[0] - last[0], p[1] - last[1]) >= step:
                    out.append((tag, ph, p)); last = p
            out.append((tag, ph, path[-1]))
    return out


def main(var):
    coupon = var == "coupon"                         # the fit-test coupon uses the Y variant's latch geometry
    v = L.VARIANTS["Y" if coupon else var]
    rec = load("Latch_test_base" if coupon else f"Receiver_{var}")
    fin = load("Latch_test_key" if coupon else f"Finger_{var}_flat")
    sh = load("Latch_shuttle")
    print(f"=== variant {var}: volumes receiver {V(rec):.3f}, finger {V(fin):.3f}, shuttle {V(sh):.3f} mm3")
    # ---- shuttle in its channel over its whole travel
    worst = 0.0
    for k in range(21):
        u = L.SH_U0 + (L.SH_U1 - L.SH_U0) * k / 20
        worst = max(worst, overlap(shuttle_at(sh, v, u), rec))
    print(f"shuttle sliding over its full travel u {L.SH_U0:.2f}..{L.SH_U1:.2f}: max overlap with receiver {worst:.6f} mm3")
    # ---- 0. approach: finger coming in from 25 mm away up to the funnel mouth, shuttle parked anywhere in its travel
    body, jaw = solids(os.path.join(OUT, "MHZL2-16D2_body.STEP"))[0], solids(os.path.join(OUT, "MHZL2-16D2_jaw.STEP"))[0]
    s_mouth = L.D_MOUTH - L.D_C
    worst = {"finger x shuttle": 0.0, "finger x receiver": 0.0, "finger x SMC body/jaw": 0.0}
    n = 0
    for k in range(51):
        s = -25.0 + (s_mouth + 25.0) * k / 50
        F = finger_at(fin, v, s)
        worst["finger x receiver"] = max(worst["finger x receiver"], overlap(F, rec))
        if not coupon:
            worst["finger x SMC body/jaw"] = max(worst["finger x SMC body/jaw"], overlap(F, body), overlap(F, jaw))
        for u in (L.SH_U0, 0.5 * (L.SH_U0 + L.SH_U1), L.SH_U1):
            worst["finger x shuttle"] = max(worst["finger x shuttle"], overlap(F, shuttle_at(sh, v, u))); n += 1
    print(f"approach (finger from 25 mm out to the mouth, shuttle at both ends and middle, {n} states): worst overlaps "
          + ", ".join(f"{k} {x:.6f}" for k, x in worst.items()) + " mm3")
    approach_bad = max(worst.values()) > TOL
    # ---- 1. cycle
    samples = cycle_samples()
    names = ["insert/push 1", "release 1 (latch)", "push 2", "release 2 (out)"]
    bad = 0; worst = {"finger x shuttle": 0.0, "finger x receiver": 0.0, "shuttle x receiver": 0.0}
    min_gap = 1e9
    for tag, ph, (u, d) in samples:
        s = d - L.D_C
        F, Sh = finger_at(fin, v, s), shuttle_at(sh, v, u)
        o = {"finger x shuttle": overlap(F, Sh), "finger x receiver": overlap(F, rec), "shuttle x receiver": overlap(Sh, rec)}
        for kk, val in o.items():
            worst[kk] = max(worst[kk], val)
        if max(o.values()) > TOL:
            bad += 1
            if bad <= 5:
                print(f"   COLLISION {tag} {names[ph]} at u {u:.3f} d {d:.3f}: " + ", ".join(f"{k} {x:.4f}" for k, x in o.items()))
        min_gap = min(min_gap, gap(F, Sh))
    print(f"cycle: {len(samples)} positions (two start positions, all four strokes): "
          f"{'no collisions' if not bad else str(bad) + ' with collisions'}; worst overlaps " +
          ", ".join(f"{k} {x:.6f}" for k, x in worst.items()) + f" mm3; closest peg-to-finger {min_gap:.3f} mm")
    # ---- 2. retention
    F0, S0 = finger_at(fin, v, 0.0), shuttle_at(sh, v, L.U_C)
    print(f"latched: peg-to-finger gap {gap(F0, S0):.3f} mm (design clearance {TG.CLR})")
    for label, F, Sh in (("finger pulled 0.3 mm outward", finger_at(fin, v, -0.3), S0),
                         ("finger pulled 1.0 mm outward", finger_at(fin, v, -1.0), S0),
                         ("shuttle pushed 0.3 mm to +u", F0, shuttle_at(sh, v, L.U_C + 0.3)),
                         ("shuttle pushed 0.3 mm to -u", F0, shuttle_at(sh, v, L.U_C - 0.3))):
        o = overlap(F, Sh)
        print(f"   {label:32s}: overlap {o:.4f} mm3 -> {'BLOCKED (holds)' if o > TOL else 'FREE (does not hold!)'}")
    # bottom of the push: the hard stop must not be reached before the track bottom
    Fb = finger_at(fin, v, L.OVERTRAVEL + L.STOP_MARGIN + 0.05)
    print(f"finger {L.STOP_MARGIN + 0.05:.2f} mm past the track bottom: overlap with receiver {overlap(Fb, rec):.4f} mm3 "
          f"(> 0 = hard stop engaged); at the track bottom: {overlap(finger_at(fin, v, L.OVERTRAVEL), rec):.6f}")
    # ---- 3. gripper states
    total = int(approach_bad)
    if coupon:
        return bad, total
    for closing, label in ((0.0, "FULL OPEN"), (1.0, "holding a 10 mm cube"), (6.0, "FULL CLOSED")):
        A = [jaw, rec, F0, S0]; B = [rotz180(x) for x in A]
        A = [tr(x, (-closing, 0, 0)) for x in A]; B = [tr(x, (closing, 0, 0)) for x in B]
        parts = [("body", body)] + list(zip(["jaw A", "receiver A", "finger A", "shuttle A"], A)) + \
                list(zip(["jaw B", "receiver B", "finger B", "shuttle B"], B))
        coll = []
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                (na, a), (nb, b) = parts[i], parts[j]
                if {na, nb} == {"jaw A", "jaw B"} or (na == "body" and nb.startswith("jaw")):
                    continue
                o = overlap(a, b)
                if o > TOL:
                    coll.append((na, nb, o))
        gfa = gap(parts[3][1], parts[7][1])
        print(f"gripper {label:22s}: {'no collisions' if not coll else coll}; finger-to-finger gap {gfa:.3f} mm")
        total += len(coll)
    return bad, total


if __name__ == "__main__":
    var = sys.argv[1] if len(sys.argv) > 1 else "Y"
    bad, total = main(var)
    print("RESULT:", "PASS" if not bad and not total else "FAIL")
