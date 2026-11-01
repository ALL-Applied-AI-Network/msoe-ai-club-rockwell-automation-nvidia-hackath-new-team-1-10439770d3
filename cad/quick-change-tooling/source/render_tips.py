"""render_tips.py: catalogue contact sheet of the tip fingers -> out/tips_catalog_<v>.png

    python render_tips.py [Y|Z] [tip ...]
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
from finger_tips import TIPS

LIGHT = np.array([0.5, -0.6, 0.65]); LIGHT /= np.linalg.norm(LIGHT)


def draw(ax, name, title, zmin=112.0):
    m = to_mesh(C.load(name), 0.05)
    T = np.asarray(m.triangles); N = np.asarray(m.face_normals)
    keep = T[:, :, 2].min(axis=1) > zmin
    T, N = T[keep], N[keep]
    base = np.array([235, 130, 40]) / 255
    col = np.clip(base[None, :] * (0.38 + 0.62 * np.abs(N @ LIGHT))[:, None], 0, 1)
    ax.add_collection3d(Poly3DCollection(T, facecolors=col, edgecolors="none"))
    c = np.array([14.0, 0.0, 136.0]); r = 23.0
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] + r, c[2] - r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(-24, 215); ax.set_axis_off()
    ax.set_title(title, fontsize=10, weight="bold")


def main(v, tips):
    n = len(tips); cols = 5; rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(cols * 4.0, rows * 4.3), dpi=85)
    for k, t in enumerate(tips):
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d", computed_zorder=False)
        draw(ax, f"Finger_{v}_{t}", f"{k + 1}. {t}")
        fig.text((k % cols + 0.5) / cols, 1 - (k // cols + 0.97) / rows, TIPS[t][1][:58], ha="center", fontsize=7.5,
                 wrap=True)
    fig.suptitle(f"Finger tips, variant {v}: same latch, 20 tips (gripping face at the left of each view)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(C.OUT, f"tips_catalog_{v}.png")
    fig.savefig(out, facecolor="white"); print("wrote", out)


if __name__ == "__main__":
    a = sys.argv[1:]
    v = next((x for x in a if x in ("Y", "Z")), "Z")
    main(v, [x for x in a if x in TIPS] or list(TIPS))
