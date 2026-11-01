"""render_latch.py: preview PNGs from the SolidWorks STEP exports (matplotlib, flat-shaded, offscreen).

out/preview_Y.png, preview_Z.png      the gripper with both latched fingers
out/preview_exploded_Y.png            receiver, shuttle and finger pulled apart along the insertion direction
out/preview_track_Y.png               the finger's tongue face, track facing the viewer, with the shuttle peg latched
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0, os.environ.get("ATEAM_CAD_HELPERS", os.path.dirname(os.path.abspath(__file__))))
from occ_mesh import to_mesh
import check_latch as C
import latch_layout as L

OUT = C.OUT
COL = dict(body=(170, 178, 195), jaw=(150, 150, 160), receiver=(70, 76, 86), shuttle=(205, 30, 40), finger=(235, 130, 40))
LIGHT = np.array([0.5, -0.6, 0.7]); LIGHT /= np.linalg.norm(LIGHT)


def draw(ax, items, elev, azim, zoom=1.0):
    tris, cols = [], []
    for shape, rgb in items:
        m = to_mesh(shape, 0.04)
        base = np.array(rgb) / 255
        shade = 0.40 + 0.60 * np.abs(m.face_normals @ LIGHT)
        tris.append(m.triangles); cols.append(np.clip(base[None, :] * shade[:, None], 0, 1))
    T = np.vstack(tris); Cc = np.vstack(cols)
    ax.add_collection3d(Poly3DCollection(T, facecolors=Cc, edgecolors="none", linewidths=0))
    lo, hi = T.reshape(-1, 3).min(0), T.reshape(-1, 3).max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2 / zoom
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev, azim); ax.set_axis_off()


def save(items, fn, elev, azim, title, zoom=1.0):
    fig = plt.figure(figsize=(9, 8), dpi=120)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    draw(ax, items, elev, azim, zoom)
    fig.text(0.02, 0.97, title, fontsize=11, va="top")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(os.path.join(OUT, fn), facecolor="white"); plt.close(fig)
    print("wrote", fn)


def gripper(var):
    v = L.VARIANTS[var]
    body, jaw = C.load("MHZL2-16D2_body"), C.load("MHZL2-16D2_jaw")
    rec, fin, sh = C.load(f"Receiver_{var}"), C.load(f"Finger_{var}_flat"), C.load("Latch_shuttle")
    S0 = C.shuttle_at(sh, v, L.U_C)
    A = [(jaw, COL["jaw"]), (rec, COL["receiver"]), (S0, COL["shuttle"]), (fin, COL["finger"])]
    return [(body, COL["body"])] + A + [(C.rotz180(s), c) for s, c in A], (rec, fin, sh, S0)


def track_section(var):
    """slice the SolidWorks finger 1 mm under the tongue face (through the track) and overlay the designed track
    outline, the pin-centre limits and the peg latched in the notch"""
    import track_geom as TG
    v = L.VARIANTS[var]
    fin = C.load(f"Finger_{var}_flat")
    m = to_mesh(fin, 0.01)
    x_cut = L.X_TF - 1.0
    sec = m.section(plane_origin=[x_cut, 0, 0], plane_normal=[1, 0, 0])
    # plot in (horizontal, vertical) = (world Y, world Z)
    fig, ax = plt.subplots(figsize=(8, 9), dpi=120)
    for e in sec.discrete:
        ax.plot(e[:, 1], e[:, 2], color="black", lw=1.0)
    W = lambda q: v.track_to_world(q[0], q[1])
    o = [W(q) for q in TG.elements_to_points(TG.outer_outline())]; o.append(o[0])
    ax.plot([p[1] for p in o], [p[2] for p in o], color="tab:blue", lw=0.8, ls="--", label="designed track (exact)")
    isl = [W(q) for q in TG.ISLAND]; isl.append(isl[0])
    ax.plot([p[1] for p in isl], [p[2] for p in isl], color="tab:blue", lw=0.8, ls="--")
    F = TG.free_polygon()
    for ring in [F.exterior] + list(F.interiors):
        pts = [W(q) for q in ring.coords]
        ax.plot([p[1] for p in pts], [p[2] for p in pts], color="gray", lw=0.6, ls=":")
    pc = W((L.U_C, L.D_C))
    ax.add_patch(plt.Circle((pc[1], pc[2]), L.PIN_D / 2, color="tab:red", alpha=0.6, label="peg, latched"))
    t0 = W((0.0, L.D_MOUTH))
    ax.annotate("mouth (finger enters here)", (t0[1], t0[2]), fontsize=9, color="tab:blue")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("world Y (mm)"); ax.set_ylabel("world Z (mm)")
    if var == "Y":
        ax.invert_yaxis()
    ax.set_title(f"Finger_{var}: SolidWorks solid sliced at X = {x_cut:.2f} (black) vs designed track (blue)")
    lo = min(p[1] for p in o) - 3, max(p[1] for p in o) + 3
    ax.set_xlim(*lo); zs = [p[2] for p in o]; ax.set_ylim(max(zs) + 3, min(zs) - 3) if var == "Y" else ax.set_ylim(min(zs) - 3, max(zs) + 3)
    ax.legend(loc="lower left", fontsize=8)
    fn = f"preview_track_{var}.png"; fig.tight_layout(); fig.savefig(os.path.join(OUT, fn)); plt.close(fig)
    print("wrote", fn)


if __name__ == "__main__":
    # tool axis Z points away from the robot: show the fingertips toward the viewer, from below-front
    items, (rec, fin, sh, S0) = gripper("Y")
    save(items, "preview_Y.png", -25, -55, "Variant Y (finger slides on sideways): both fingers latched, gripper open")
    items_z, _ = gripper("Z")
    save(items_z, "preview_Z.png", -25, -55, "Variant Z (finger slides up along its length): both fingers latched")
    v = L.VarY
    ex = [(rec, COL["receiver"]), (C.tr(S0, (6.0, 0, 0)), COL["shuttle"]), (C.finger_at(fin, v, -28.0), COL["finger"])]
    save(ex, "preview_exploded_Y.png", -20, -35,
         "Variant Y exploded: receiver (bolts to the SMC finger), shuttle (pulled out of its slot), finger (slides in +Y)")
    track_section("Y"); track_section("Z")
