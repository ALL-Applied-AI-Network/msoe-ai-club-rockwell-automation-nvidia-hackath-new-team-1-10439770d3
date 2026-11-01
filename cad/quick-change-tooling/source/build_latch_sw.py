"""build_latch_sw.py: native SolidWorks parts and assemblies for the push-push finger mount.

    python build_latch_sw.py [job ...]
    jobs: shuttle recY finY finY_V recZ finZ finZ_V coupon asmY asmZ tipsY tipsZ tipY:<tip> tipZ:<tip>
          (default: all; the 20 tips are in finger_tips.py)

Every part is built in the gripper frame at the latched position of the +X jaw (see latch_layout.py), so the
assemblies place the +X side with identity transforms and the -X side rotated 180 deg about Z. Writes out/*.SLDPRT,
out/*.STEP (for the checks) and out/*.STL (for printing).
"""
import math, os, sys, shutil
sys.path.insert(0, os.environ.get("ATEAM_CAD_HELPERS", os.path.dirname(os.path.abspath(__file__))))
from swkit2 import GPart, g, app, MM, set_colour
import track_geom as TG
import latch_layout as L
from latch_layout import VarY, VarZ
from finger_tips import TIPS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
GRIPPER_OUT = os.path.join(os.path.dirname(HERE), "out")
COL = dict(receiver=(70, 76, 86), finger=(235, 130, 40), shuttle=(200, 30, 40), coupon=(60, 140, 200))
TAN = math.tan(L.DOVE)


# ======================================================================== helpers
def normal(p):
    return (p.R[0][2], p.R[1][2], p.R[2][2])


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


AX = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0), "-Y": (0, -1, 0), "+Z": (0, 0, 1), "-Z": (0, 0, -1)}


def sk(p, ax, off):
    p.sketch(ax, off); p.ax = ax
    return p


def boss(p, depth, label, d):
    """extrude the open sketch by depth toward world direction d ('+X', ...)"""
    n = normal(p); name = p.done()
    v0 = p.volume()
    from swkit import select, clear
    clear(p.m); select(p.m, name, "SKETCH")
    # FeatureExtrusion3(Sd, Flip, Dir, ...): Dir (3rd) reverses the direction; swkit's Part.boss puts it in Flip
    f = g(p.fm, "FeatureExtrusion3", True, False, dot(n, AX[d]) < 0, 0, 0, depth * MM, 0.0, False, False,
          False, False, 0.0, 0.0, False, False, False, False, True, True, True, 0, 0.0, False)
    p._named(f, label, "boss")
    assert p.volume() > v0 + 1e-6, f"boss {label} added nothing"
    return f


def cut(p, depth, label, d):
    n = normal(p); name = p.done()
    return p.cut(name, depth, label, along=+1 if dot(n, AX[d]) > 0 else -1)


def rect(p, a, b):
    p.rect(a, b)


def hexagon_x(p, x, y0, z0, af):
    r = af / math.sqrt(3)
    p.poly([(x, y0 + r * math.cos(math.radians(60 * k)), z0 + r * math.sin(math.radians(60 * k))) for k in range(6)])


def trapezoid_xz(p, y, zc, hw0, hw1, x0, x1):
    """dovetail profile in the plane Y = y: half-width hw0 at X = x0, hw1 at X = x1"""
    p.poly([(x0, y, zc - hw0), (x1, y, zc - hw1), (x1, y, zc + hw1), (x0, y, zc + hw0)])


def trapezoid_xy(p, z, yc, hw0, hw1, x0, x1):
    p.poly([(x0, yc - hw0, z), (x1, yc - hw1, z), (x1, yc + hw1, z), (x0, yc + hw0, z)])


def save(p, rgb, stl=True):
    set_colour(p.m, rgb)
    path = os.path.join(OUT, p.name + ".SLDPRT")
    r1 = g(p.m, "SaveAs3", path, 0, 2)
    r2 = g(p.m, "SaveAs3", os.path.join(OUT, p.name + ".STEP"), 0, 2)
    r3 = g(p.m, "SaveAs3", os.path.join(OUT, p.name + ".STL"), 0, 2) if stl else "-"
    print(f"  saved {p.name}: sldprt {r1} step {r2} stl {r3}  volume {p.volume():.3f} mm3", flush=True)
    app().CloseDoc(g(p.m, "GetTitle"))


# ======================================================================== shared features
def sleeve(p):
    """the SMC side, identical to the verified Finger_base_adapter sleeve"""
    f = L.FINGER
    sk(p, "Z", L.SLEEVE_Z0); rect(p, (L.X_IN, -L.Y_SL, L.SLEEVE_Z0), (L.X_OUT, L.Y_SL, L.SLEEVE_Z0))
    boss(p, L.SLEEVE_Z1 - L.SLEEVE_Z0, "Sleeve (SMC finger)", "+Z")


def smc_interface(p):
    f = L.FINGER
    sk(p, "Z", L.SLEEVE_Z0)
    rect(p, (f["x0"] - L.CLR, -f["y"] - L.CLR, L.SLEEVE_Z0), (f["x1"] + L.CLR, f["y"] + L.CLR, L.SLEEVE_Z0))
    cut(p, L.POCKET_TOP - L.SLEEVE_Z0, "SMC finger pocket", "+Z")
    sk(p, "X", L.X_IN)
    for z in f["holes"]:
        p.circle((L.X_IN, 0.0, z), L.M3_CLR / 2)
    cut(p, L.X_OUT - L.X_IN, "M3 cross bolt holes (SMC finger)", "+X")
    sk(p, "X", L.X_IN)
    for z in f["holes"]:
        hexagon_x(p, L.X_IN, 0.0, z, L.M3_NUT_AF)
    cut(p, L.NUT_DEPTH, "M3 nut traps (inner face)", "+X")


def track_cut(p, v, label="Heart track"):
    """the push-push track on the tongue face, exact lines and arcs from track_geom"""
    sk(p, "X", L.X_TF)
    W = lambda q: v.track_to_world(q[0], q[1])
    for e in TG.outer_outline():
        if e[0] == "line":
            p.line(W(e[1]), W(e[2]))
        else:
            _, c, a, b, mid = e
            p.arc_mid(W(c), W(a), W(b), W(mid))
    for e in TG.island_outline():                      # island with filleted corners
        if e[0] == "line":
            p.line(W(e[1]), W(e[2]))
        else:
            _, c, a, b, mid = e
            p.arc_mid(W(c), W(a), W(b), W(mid))
    cut(p, L.TRACK_DEPTH, label, "-X")


# ======================================================================== shuttle
def shuttle():
    """built in its own frame: peg axis on X (absolute X as in the gripper frame), centred at Y = Z = 0; head long
    along Z (= u, the direction it slides), wide along Y (= d)"""
    p = GPart("Latch_shuttle")
    sk(p, "X", L.HEAD_X0)
    rect(p, (L.HEAD_X0, -L.HEAD_W / 2, -L.HEAD_L / 2), (L.HEAD_X0, L.HEAD_W / 2, L.HEAD_L / 2))
    boss(p, L.HEAD_T, "Head (rides under the T-slot lips)", "+X")
    sk(p, "X", L.HEAD_X0); p.circle((L.HEAD_X0, 0.0, 0.0), L.PIN_D / 2)
    boss(p, L.PEG_LEN, "Peg (follows the heart track)", "-X")
    save(p, COL["shuttle"])


# ======================================================================== variant Y
def receiver_Y(coupon=False):
    v = VarY
    name = "Latch_test_base" if coupon else "Receiver_Y"
    p = GPart(name)
    z0 = v.ZC - L.G_FLOOR_HW - 1.5 if coupon else L.POCKET_TOP
    x_out = L.X_CHAN + 1.5 if coupon else L.X_OUT
    if not coupon:
        sleeve(p)
    sk(p, "X", L.X_I); rect(p, (L.X_I, v.Y_TR, z0), (L.X_I, v.Y_EW0, v.Z_B))
    boss(p, x_out - L.X_I, "Lap (carries the dovetail groove and the latch)", "+X")
    sk(p, "Y", v.Y_EW0); rect(p, (L.X_IN, v.Y_EW0, z0), (x_out, v.Y_EW0, v.Z_B))
    boss(p, v.END_WALL_T, "End wall (stop + eject spring)", "+Y")
    if not coupon:
        smc_interface(p)
    # dovetail groove along Y, open at the trailing (-Y) end
    a = L.G_NECK_HW - 0.1 * TAN
    sk(p, "Y", v.Y_EW0); trapezoid_xz(p, v.Y_EW0, v.ZC, a, L.G_FLOOR_HW, L.X_I - 0.1, L.X_GF)
    cut(p, v.Y_EW0 - v.Y_TR + 1.0, "Dovetail groove", "-Y")
    # T-slot for the shuttle, along Z (= u), open through the lap bottom for assembly
    z_pin_lo = v.Z_P + (L.SH_U0 - U_C())
    sk(p, "Z", v.Z_B); rect(p, (L.X_I - 0.1, v.Y_P - L.NECK_W / 2, v.Z_B), (L.X_LIP, v.Y_P + L.NECK_W / 2, v.Z_B))
    cut(p, v.Z_B - (z_pin_lo - L.NECK_W / 2), "Shuttle slot (peg)", "-Z")
    sk(p, "Z", v.Z_B); rect(p, (L.X_LIP, v.Y_P - L.CHAN_W / 2, v.Z_B), (L.X_CHAN, v.Y_P + L.CHAN_W / 2, v.Z_B))
    cut(p, v.Z_B - (z_pin_lo - L.HEAD_L / 2 - 0.2), "Shuttle channel (head)", "-Z")
    z_ret = v.Z_P + (L.SH_U1 - U_C()) + L.HEAD_L / 2 + 0.2 + 1.0
    sk(p, "X", x_out); p.circle((x_out, v.Y_P, z_ret), L.M2_PILOT / 2)
    cut(p, x_out - (L.X_LIP - 0.7), "M2 retaining screw (keeps the shuttle in)", "-X")
    # eject spring bore in the end wall, last mm undersize so the spring's end coil presses in
    rb, rs, dep = L.SPRING_BORE_D / 2, L.SPRING_SEAT_D / 2, L.SPRING_BORE_DEPTH
    p.hole("Eject spring bore", "Y", (v.SPRING_AT[0], v.Y_EW0, v.SPRING_AT[1]),
           [(0.0, rb), (dep - 1.0, rb), (dep - 1.0, rs), (dep, rs)], v.SPRING_AT[1], depth_dir=+1)
    save(p, COL["coupon" if coupon else "receiver"])


def U_C():
    return L.U_C


def finger_Y(tip="flat", coupon=False):
    v = VarY
    name = "Latch_test_key" if coupon else f"Finger_Y_{tip}"
    p = GPart(name)
    z0 = v.ZC - L.FACE_HW - 0.4
    z1 = v.ZC + L.FACE_HW + 0.4 if coupon else v.ROOT_Z1
    y_tr = v.Y_TR - (6.0 if coupon else 0.0)            # coupon: a handle sticks out of the trailing end
    sk(p, "X", L.FINGER_IN); rect(p, (L.FINGER_IN, y_tr, z0 if coupon else v.ROOT_Z0), (L.FINGER_IN, v.Y_LE, z1))
    boss(p, L.X_I - L.FINGER_IN, "Root", "+X")
    a = L.NECK_HW - 0.1 * TAN
    sk(p, "Y", v.Y_LE); trapezoid_xz(p, v.Y_LE, v.ZC, a, L.FACE_HW, L.X_I - 0.1, L.X_TF)
    boss(p, v.Y_LE - v.Y_TR, "Dovetail tongue", "-Y")
    track_cut(p, v)
    if not coupon:
        beam_and_tip(p, v.ROOT_Z1, tip)
    save(p, COL["coupon" if coupon else "finger"])


def beam_and_tip(p, z_beam, tip):
    sk(p, "Z", z_beam); rect(p, (L.FINGER_IN, -L.BEAM_Y, z_beam), (L.BEAM_X1, L.BEAM_Y, z_beam))
    boss(p, L.TIP_END - z_beam, "Beam", "+Z")
    if tip in TIPS:                                    # one of the 20 tool / gripping tips (finger_tips.py)
        TIPS[tip][0](p)
    sk(p, "X", L.FACE_OPEN)
    holes = L.ACC_HOLES
    if holes[0] < z_beam + 2.5:                        # a long root (variant Z) leaves less beam: keep the holes
        pitch = holes[1] - holes[0]                    # clear of the root/track and of the V-groove at the tip
        holes = [z for z in (z_beam + 2.5, z_beam + 2.5 + pitch) if z + L.M3_CLR / 2 < L.TIP_END - 7.0]
    for z in holes:
        p.circle((L.FACE_OPEN, 0.0, z), L.M3_CLR / 2)
    cut(p, L.BEAM_X1 - L.FACE_OPEN, "M3 accessory / sensor holes", "+X")
    if tip == "V":
        zc, dep = L.TIP_END - 3.2, 2.5
        sk(p, "Y", 0.0)
        p.poly([(L.FACE_OPEN - 0.5, 0.0, zc - dep - 0.5), (L.FACE_OPEN + dep, 0.0, zc), (L.FACE_OPEN - 0.5, 0.0, zc + dep + 0.5)])
        p.cut(p.done(), 2 * L.BEAM_Y + 2, "90 deg V-groove (small lying cylinders)", mid=True)


# ======================================================================== variant Z
def receiver_Z():
    v = VarZ
    p = GPart("Receiver_Z")
    sleeve(p)
    sk(p, "Z", v.UPPER_Z0); rect(p, (L.X_IN, -v.HALF_W, v.UPPER_Z0), (L.X_OUT, v.HALF_W, v.UPPER_Z0))
    boss(p, v.STOP_Z - v.UPPER_Z0, "Upper block (stop face + eject spring)", "+Z")
    sk(p, "Z", v.STOP_Z); rect(p, (L.X_I, -v.HALF_W, v.STOP_Z), (L.X_OUT, v.HALF_W, v.STOP_Z))
    boss(p, v.Z_B - v.STOP_Z, "Lap (carries the dovetail groove and the latch)", "+Z")
    smc_interface(p)
    a = L.G_NECK_HW - 0.1 * TAN
    sk(p, "Z", v.Z_B); trapezoid_xy(p, v.Z_B, v.Y_C, a, L.G_FLOOR_HW, L.X_I - 0.1, L.X_GF)
    cut(p, v.Z_B - (v.STOP_Z - 0.2), "Dovetail groove", "-Z")
    y_pin_lo = v.Y_P + (L.SH_U0 - L.U_C)
    sk(p, "Y", v.HALF_W); rect(p, (L.X_I - 0.1, v.HALF_W, v.Z_P - L.NECK_W / 2), (L.X_LIP, v.HALF_W, v.Z_P + L.NECK_W / 2))
    cut(p, v.HALF_W - (y_pin_lo - L.NECK_W / 2), "Shuttle slot (peg)", "-Y")
    sk(p, "Y", v.HALF_W); rect(p, (L.X_LIP, v.HALF_W, v.Z_P - L.CHAN_W / 2), (L.X_CHAN, v.HALF_W, v.Z_P + L.CHAN_W / 2))
    cut(p, v.HALF_W - (y_pin_lo - L.HEAD_L / 2 - 0.2), "Shuttle channel (head)", "-Y")
    y_ret = v.Y_P + (L.SH_U1 - L.U_C) + L.HEAD_L / 2 + 0.2 + 1.0
    sk(p, "X", L.X_OUT); p.circle((L.X_OUT, y_ret, v.Z_P), L.M2_PILOT / 2)
    cut(p, L.X_OUT - (L.X_LIP - 0.7), "M2 retaining screw (keeps the shuttle in)", "-X")
    rb, rs, dep = L.SPRING_BORE_D / 2, L.SPRING_SEAT_D / 2, L.SPRING_BORE_DEPTH
    x, y = v.SPRING_AT
    p.hole("Eject spring bore", "Z", (x, y, v.STOP_Z), [(0.0, rb), (dep - 1.0, rb), (dep - 1.0, rs), (dep, rs)], x,
           depth_dir=-1)
    save(p, COL["receiver"])


def finger_Z(tip="flat"):
    v = VarZ
    p = GPart(f"Finger_Z_{tip}")
    sk(p, "X", L.FINGER_IN); rect(p, (L.FINGER_IN, -v.ROOT_HW, v.Z_LE), (L.FINGER_IN, v.ROOT_HW, v.Z_SH))
    boss(p, L.X_I - L.FINGER_IN, "Root", "+X")
    a = L.NECK_HW - 0.1 * TAN
    sk(p, "Z", v.Z_LE); trapezoid_xy(p, v.Z_LE, v.Y_C, a, L.FACE_HW, L.X_I - 0.1, L.X_TF)
    boss(p, v.ROOT_Z1 - v.Z_LE, "Dovetail tongue", "+Z")
    track_cut(p, v)
    beam_and_tip(p, v.Z_SH, tip)
    save(p, COL["finger"])


# ======================================================================== assemblies
def assembly(variant):
    import pythoncom
    from win32com.client import gencache, VARIANT
    v = L.VARIANTS[variant]
    for f in ("MHZL2-16D2_body", "MHZL2-16D2_jaw"):
        for ext in (".SLDPRT", ".STEP"):
            shutil.copy2(os.path.join(GRIPPER_OUT, f + ext), os.path.join(OUT, f + ext))
    I = ((1, 0, 0), (0, 1, 0), (0, 0, 1)); R = ((-1, 0, 0), (0, -1, 0), (0, 0, 1))
    sh = v.shuttle_at(L.U_C)
    if variant == "Y":
        S_rows, S_t = I, (0.0, sh[1], sh[2])            # shuttle local Z (u) -> world Z, local Y (d) -> world Y
    else:
        S_rows, S_t = ((1, 0, 0), (0, 0, 1), (0, -1, 0)), (0.0, sh[1], sh[2])   # local Z -> world Y
    def rot180(rows, t):
        rr = tuple(tuple(-x if k < 2 else x for k, x in enumerate(r)) for r in rows)
        rr = tuple((-r[0], -r[1], r[2]) for r in rows)
        return rr, (-t[0], -t[1], t[2])
    S2_rows, S2_t = rot180(S_rows, S_t)
    rec, fin = f"Receiver_{variant}", f"Finger_{variant}_flat"
    parts = [("MHZL2-16D2_body", I, (0, 0, 0)), ("MHZL2-16D2_jaw", I, (0, 0, 0)), ("MHZL2-16D2_jaw", R, (0, 0, 0)),
             (rec, I, (0, 0, 0)), (rec, R, (0, 0, 0)), ("Latch_shuttle", S_rows, S_t), ("Latch_shuttle", S2_rows, S2_t),
             (fin, I, (0, 0, 0)), (fin, R, (0, 0, 0))]
    s = app(); g(s, "CloseAllDocuments", True)
    tlb = pythoncom.LoadTypeLib(r"C:\Program Files\SOLIDWORKS Corp\SOLIDWORKS\sldworks.tlb")
    la = tlb.GetLibAttr(); mod = gencache.EnsureModule(str(la[0]), la[1], la[3], la[4])
    for f in {pp[0] for pp in parts}:
        e = VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0); w = VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
        s.OpenDoc6(os.path.join(OUT, f + ".SLDPRT"), 1, 1, "", e, w)
    tpl = g(s, "GetUserPreferenceStringValue", 9)
    if not os.path.exists(tpl or ""):
        import glob
        tpl = (glob.glob(r"C:\ProgramData\SolidWorks\SOLIDWORKS 2026\templates\**\*.asmdot", recursive=True) or [""])[0]
    a = g(s, "NewDocument", tpl, 0, 0, 0)
    mu = mod.IMathUtility(g(s, "GetMathUtility")._oleobj_)
    comps = []
    for f, rows, t in parts:
        c = g(a, "AddComponent5", os.path.join(OUT, f + ".SLDPRT"), 0, "", False, "", 0.0, 0.0, 0.0)
        arr = [float(x) for r in rows for x in r] + [t[0] * MM, t[1] * MM, t[2] * MM, 1.0, 0.0, 0.0, 0.0]
        tr = mu.CreateTransform(VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, arr))
        comp = mod.IComponent2(c._oleobj_)
        comp.Transform2 = mod.IMathTransform(tr._oleobj_)
        got = [round(x, 6) for x in comp.Transform2.ArrayData[:12]]
        ok = got[:9] == [float(x) for r in rows for x in r] and all(abs(got[9 + k] - t[k] * MM) < 1e-9 for k in range(3))
        print(f"  + {f:22s} {t}  ok={ok}")
        comps.append(c)
    g(a, "ClearSelection2", True)
    from swkit import NOTHING
    for c in comps:
        g(c, "Select4", True, NOTHING, False)
    g(a, "FixComponent"); g(a, "ForceRebuild3", False)
    nm = f"Gripper_latch_{variant}"
    for ext in ("SLDASM", "STEP"):
        print(f"  saved {nm}.{ext}:", g(a, "SaveAs3", os.path.join(OUT, f"{nm}.{ext}"), 0, 2))
    g(s, "CloseAllDocuments", True)


JOBS = {"shuttle": shuttle, "recY": receiver_Y, "finY": lambda: finger_Y("flat"), "finY_V": lambda: finger_Y("V"),
        "recZ": receiver_Z, "finZ": lambda: finger_Z("flat"), "finZ_V": lambda: finger_Z("V"),
        "coupon": lambda: (receiver_Y(coupon=True), finger_Y(coupon=True)),
        "asmY": lambda: assembly("Y"), "asmZ": lambda: assembly("Z"),
        "tipsY": lambda: [finger_Y(t) for t in TIPS], "tipsZ": lambda: [finger_Z(t) for t in TIPS]}
for _t in TIPS:                                        # single tips: tipY:claw, tipZ:socket8, ...
    JOBS[f"tipY:{_t}"] = (lambda t: lambda: finger_Y(t))(_t)
    JOBS[f"tipZ:{_t}"] = (lambda t: lambda: finger_Z(t))(_t)

if __name__ == "__main__":
    which = sys.argv[1:] or list(JOBS)
    g(app(), "CloseAllDocuments", True)
    for j in which:
        print(j, flush=True)
        JOBS[j]()
