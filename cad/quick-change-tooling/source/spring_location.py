"""spring_location.py: cut-away views showing where the eject spring sits (out/spring_location.png).

Left: variant Y, cut horizontally through the spring axis (plane Z = spring height), seen from the fingertip side.
Right: variant Z, cut through the spring axis (plane Y = 6), seen from the front.
Grey = receiver, orange = finger, red = shuttle peg, blue zig-zag = the spring (drawn at its latched length).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly
sys.path.insert(0, os.environ.get("ATEAM_CAD_HELPERS", os.path.dirname(os.path.abspath(__file__))))
from occ_mesh import to_mesh
import check_latch as C
import latch_layout as L

OUT = C.OUT
FILL = dict(receiver="#8c939e", finger="#eb8228", shuttle="#d71e28")


def section(shape, origin, normal, axes):
    m = to_mesh(shape, 0.02)
    sec = m.section(plane_origin=origin, plane_normal=normal)
    if sec is None:
        return []
    return [e[:, axes] for e in sec.discrete]


def draw_loops(ax, loops, color):
    # fill closed loops with even-odd so holes show
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    polys = [Polygon(l) for l in loops if len(l) > 3]
    if not polys:
        return
    region = polys[0]
    for p in polys[1:]:
        region = region.symmetric_difference(p)
    geoms = getattr(region, "geoms", [region])
    for g in geoms:
        if g.is_empty:
            continue
        ax.add_patch(MPoly(np.array(g.exterior.coords), closed=True, facecolor=color, edgecolor="black", lw=0.8))
        for h in g.interiors:
            ax.add_patch(MPoly(np.array(h.coords), closed=True, facecolor="white", edgecolor="black", lw=0.8))


def spring(ax, a, b, od, n=9):
    """zig-zag from point a to point b (2D), outer diameter od"""
    a, b = np.array(a, float), np.array(b, float)
    t = (b - a) / np.linalg.norm(b - a); nrm = np.array([-t[1], t[0]])
    pts = [a]
    for k in range(1, 2 * n):
        pts.append(a + (b - a) * k / (2 * n) + nrm * (od / 2) * (1 if k % 2 else -1))
    pts.append(b)
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], color="#0050c8", lw=2.2, zorder=5)


def main():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 9), dpi=100)
    # ---------------- variant Y: plane Z = spring height, axes (Y, X)
    v = L.VarY
    zc = v.SPRING_AT[1]
    rec, fin, sh = C.load("Receiver_Y"), C.load("Finger_Y_flat"), C.load("Latch_shuttle")
    S0 = C.shuttle_at(sh, v, L.U_C)
    for shape, col in ((rec, FILL["receiver"]), (fin, FILL["finger"])):
        draw_loops(a1, section(shape, [0, 0, zc], [0, 0, 1], [1, 0]), col)
    x_s = v.SPRING_AT[0]
    y0, y1 = v.Y_LE, v.Y_EW0 + L.SPRING_BORE_DEPTH            # finger leading face -> bottom of the bore
    spring(a1, (y0, x_s), (y1, x_s), L.SPRING["od"])
    a1.annotate("SPRING\n(sits in the hole in the\nreceiver's end wall,\npushes on the finger's end)",
                xy=(0.5 * (y0 + y1), x_s + 1.6), xytext=(y1 + 2.0, x_s + 9.0), fontsize=11, color="#0050c8",
                weight="bold", arrowprops=dict(arrowstyle="->", color="#0050c8", lw=1.5))
    a1.annotate("receiver end wall\n(bolted part)", xy=(v.Y_EW0 + 4.0, 18.0), xytext=(v.Y_EW0 + 2.0, 25.0), fontsize=10,
                arrowprops=dict(arrowstyle="->"))
    a1.annotate("finger root", xy=(v.Y_LE - 6.0, 9.0), xytext=(v.Y_LE - 14.0, 1.0), fontsize=10,
                arrowprops=dict(arrowstyle="->"))
    a1.annotate("finger slides in this way  ->", xy=(v.Y_TR, 4.2), fontsize=10, color="#5a2d00")
    a1.set_title(f"Variant Y: cut at Z = {zc:.1f} through the spring (looking from the fingertip end)", fontsize=12)
    a1.set_xlabel("Y (mm)  - the insertion direction"); a1.set_ylabel("X (mm)  - jaw axis; gripping face at X 6")
    a1.set_aspect("equal"); a1.set_xlim(v.Y_TR - 2, v.HALF_SPAN + 14); a1.set_ylim(3, 28); a1.grid(alpha=0.25)
    # ---------------- variant Z: plane Y = 6 (spring axis), axes (X, Z)
    w = L.VarZ
    ys = w.SPRING_AT[1]
    recz, finz = C.load("Receiver_Z"), C.load("Finger_Z_flat")
    for shape, col in ((recz, FILL["receiver"]), (finz, FILL["finger"])):
        draw_loops(a2, section(shape, [0, ys, 0], [0, 1, 0], [0, 2]), col)
    xz = w.SPRING_AT[0]
    z0, z1 = w.Z_LE, w.STOP_Z - L.SPRING_BORE_DEPTH            # finger top -> bottom of the bore (above)
    spring(a2, (xz, z0), (xz, z1), L.SPRING["od"])
    a2.annotate("SPRING\n(sits in the hole above the\nfinger, pushes the finger down)", xy=(xz + 1.6, 0.5 * (z0 + z1)),
                xytext=(xz + 12.0, z1 - 4.0), fontsize=11, color="#0050c8", weight="bold",
                arrowprops=dict(arrowstyle="->", color="#0050c8", lw=1.5))
    a2.annotate("finger slides up\nthis way", xy=(8.0, w.ROOT_Z1 - 2), xytext=(-2.0, w.ROOT_Z1 + 8), fontsize=10,
                color="#5a2d00", arrowprops=dict(arrowstyle="->", color="#5a2d00"))
    a2.set_title(f"Variant Z: cut at Y = {ys:.1f} through the spring (looking from the front)", fontsize=12)
    a2.set_xlabel("X (mm)"); a2.set_ylabel("Z (mm)  - tool axis, fingertips down")
    a2.set_aspect("equal"); a2.set_xlim(-4, 30); a2.set_ylim(w.ROOT_Z1 + 14, 55); a2.grid(alpha=0.25)
    fig.suptitle("Where the eject spring goes: grey = receiver (bolted to the SMC jaw), orange = finger, "
                 f"blue = spring (drawn latched, about {L.SPRING_BORE_DEPTH + L.OVERTRAVEL:.1f} mm long)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "spring_location.png"), facecolor="white")
    print("wrote spring_location.png")


if __name__ == "__main__":
    main()
