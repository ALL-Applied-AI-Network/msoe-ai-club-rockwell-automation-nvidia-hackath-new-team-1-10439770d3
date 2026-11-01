"""attach_demo.py: how a finger goes on and comes off (variant Y, +X jaw), from the SolidWorks STEP exports.

out/attach_storyboard.png   6 steps: line up, slide in, push to the stop, let go (latched), push again, let go + slide off
out/attach_demo.gif         the same as an animation
Left of each frame: the gripper (the -X jaw's parts hidden for clarity). Right: the heart track in the finger with the
peg's position at that moment (the pin path is the one simulate.py computes).
"""
import os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0, os.environ.get("ATEAM_CAD_HELPERS", os.path.dirname(os.path.abspath(__file__))))
from occ_mesh import to_mesh
import check_latch as C
import latch_layout as L
import track_geom as TG
import simulate as S

v = L.VarY
OUT = C.OUT
LIGHT = np.array([0.45, -0.55, 0.7]); LIGHT /= np.linalg.norm(LIGHT)
COL = dict(body=(175, 182, 198), jaw=(150, 150, 160), receiver=(70, 76, 86), shuttle=(215, 30, 40),
           finger=(235, 130, 40))


def mesh(shape, tol=0.08):
    m = to_mesh(shape, tol)
    return np.asarray(m.triangles), np.asarray(m.face_normals)


body, jaw = C.load("MHZL2-16D2_body"), C.load("MHZL2-16D2_jaw")
rec, fin, sh = C.load("Receiver_Y"), C.load("Finger_Y_flat"), C.load("Latch_shuttle")
STATIC = [mesh(body, 0.15) + (COL["body"],), mesh(jaw) + (COL["jaw"],), mesh(rec) + (COL["receiver"],)]
FIN = mesh(fin) + (COL["finger"],)
SH0 = mesh(C.shuttle_at(sh, v, 0.0), 0.03) + (COL["shuttle"],)


def shade(tris, normals, rgb):
    base = np.array(rgb) / 255
    s = 0.42 + 0.58 * np.abs(normals @ LIGHT)
    return np.clip(base[None, :] * s[:, None], 0, 1)


ARROW = {"1": (+1, "slide in"), "2": (+1, "push"), "3": (-1, "spring"), "4": (+1, "push"), "5": (-1, "spring"),
         "6": (-1, "slide off")}


def draw3d(ax, s, u, step=None):
    tris, cols = [], []
    for T, N, c in STATIC:
        tris.append(T); cols.append(shade(T, N, c))
    T, N, c = FIN; tris.append(T + np.array(v.ins) * s); cols.append(shade(T, N, c))
    T, N, c = SH0; tris.append(T + np.array([0.0, 0.0, u])); cols.append(shade(T, N, c))   # built at u = 0
    ax.add_collection3d(Poly3DCollection(np.vstack(tris), facecolors=np.vstack(cols), edgecolors="none"))
    if step in ARROW:
        sg, txt = ARROW[step]
        y0 = v.Y_LE + s + (2.0 if sg > 0 else -12.0)
        ax.quiver(4.0, y0 - 8 * sg * (sg > 0), 112.0, 0, 8.0 * sg, 0, color="#0050c8", lw=3, arrow_length_ratio=0.35)
        ax.text(4.0, y0, 118.0, txt, color="#0050c8", fontsize=11, weight="bold")
    c = np.array([14.0, -6.0, 88.0]); r = 30.0
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] + r, c[2] - r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(18, -118); ax.set_axis_off()


TRACK = TG.track_polygon()


def draw_track(ax, u, d):
    """the finger's tongue face (orange) with the track cut into it (grey), in finger coordinates
    (Y across the page, Z down); the peg (red) at track point (u, d)"""
    W = lambda q: v.track_to_world(q[0], q[1])
    rings = [TRACK.exterior] + list(TRACK.interiors)
    ax.add_patch(plt.Rectangle((v.Y_TR, v.ZC - L.FACE_HW), v.Y_LE - v.Y_TR, 2 * L.FACE_HW, color="#eb8228", zorder=0))
    for k, ring in enumerate(rings):
        pts = np.array([W(q) for q in ring.coords])
        ax.fill(pts[:, 1], pts[:, 2], color="#d9d9d9" if k == 0 else "#eb8228", zorder=1)
        ax.plot(pts[:, 1], pts[:, 2], color="#5a2d00", lw=1, zorder=2)
    p = W((u, d))
    ax.add_patch(plt.Circle((p[1], p[2]), L.PIN_D / 2, color="#d71e28", zorder=3))
    ax.annotate("finger enters this way", xy=(v.Y_LE + 1.0, v.ZC + L.FACE_HW + 1.2), fontsize=7, color="#5a2d00")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(v.Y_TR - 1.5, v.Y_LE + 9.0); ax.set_ylim(v.ZC + L.FACE_HW + 2.5, v.ZC - L.FACE_HW - 1.5)
    ax.set_title("inside the latch (finger's tongue face): red = peg", fontsize=9)


def sequence():
    """(label, finger shift s, peg u, peg d) frames: approach, push 1, release 1, push 2, release 2, pull off"""
    D = TG.derived(); sim = S.Sim(S.free_for(0.0))
    u_start = 0.5 * (L.SH_U0 + L.SH_U1)
    ok, res, paths = S.cycle(sim, u_start, 0.4, D)
    assert ok, res
    frames = []
    s_mouth = L.D_MOUTH - L.D_C
    for s in np.linspace(-34.0, s_mouth, 10):                       # finger lined up and slid toward the receiver
        frames.append(("1  Line the finger up and slide it in (+Y)", s, u_start, L.D_MOUTH + (s - s_mouth)))
    names = ["2  Push to the stop", "3  Let go: LATCHED", "4  To remove: push to the stop again",
             "5  Let go: it pops out"]
    for nm, path in zip(names, paths):
        idx = np.linspace(0, len(path) - 1, 14).astype(int)
        for i in idx:
            u, d = path[i]; frames.append((nm, d - L.D_C, u, d))
    u_end = paths[-1][-1][0]
    for s in np.linspace(s_mouth, -34.0, 8):
        frames.append(("6  Slide it off", s, u_end, L.D_MOUTH + (s - s_mouth)))
    return frames


def frame_figure(label, s, u, d, size=(11, 5.6)):
    fig = plt.figure(figsize=size, dpi=90)
    ax = fig.add_axes([0.0, 0.0, 0.62, 0.92], projection="3d", computed_zorder=False)
    draw3d(ax, s, u, label[0])
    ax2 = fig.add_axes([0.64, 0.06, 0.34, 0.82])
    draw_track(ax2, u, d)
    fig.text(0.02, 0.95, label, fontsize=13, weight="bold")
    return fig


if __name__ == "__main__":
    frames = sequence()
    # storyboard: one representative frame per step
    by = {}
    for f in frames:
        by.setdefault(f[0], []).append(f)
    keys = list(by)
    steps = [by[keys[0]][3]] + [by[k][-1] for k in keys[1:5]] + [by[keys[5]][3]]
    fig = plt.figure(figsize=(18, 17), dpi=80)
    for k, (label, s, u, d) in enumerate(steps[:6]):
        r, c = divmod(k, 2)
        ax = fig.add_axes([c * 0.5, 0.67 - r * 0.33, 0.31, 0.3], projection="3d", computed_zorder=False)
        draw3d(ax, s, u, label[0])
        ax2 = fig.add_axes([c * 0.5 + 0.31, 0.69 - r * 0.33, 0.17, 0.26]); draw_track(ax2, u, d)
        fig.text(c * 0.5 + 0.01, 0.97 - r * 0.33, label, fontsize=14, weight="bold")
    fig.savefig(os.path.join(OUT, "attach_storyboard.png"), facecolor="white"); plt.close(fig)
    print("wrote attach_storyboard.png")
    # animation
    from PIL import Image
    imgs = []
    for i, (label, s, u, d) in enumerate(frames):
        fig = frame_figure(label, s, u, d)
        fig.canvas.draw()
        imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).convert("P", palette=Image.ADAPTIVE))
        plt.close(fig)
    dur = [90] * len(imgs)
    for i, f in enumerate(frames):                     # hold on the key moments
        if i + 1 == len(frames) or frames[i + 1][0] != f[0]:
            dur[i] = 1300
    imgs[0].save(os.path.join(OUT, "attach_demo.gif"), save_all=True, append_images=imgs[1:], duration=dur, loop=0)
    print(f"wrote attach_demo.gif ({len(imgs)} frames)")
