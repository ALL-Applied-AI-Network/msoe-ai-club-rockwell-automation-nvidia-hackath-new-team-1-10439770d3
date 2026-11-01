"""render_rack.py: out/rack_loaded.png (both modules with fingers parked) and out/rack_sequence.png (drop-off moves)."""
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
import tray_rack as R

LIGHT = np.array([-0.4, -0.5, 0.75]); LIGHT /= np.linalg.norm(LIGHT)
COL = dict(rack=(60, 140, 200), finger=(235, 130, 40), other=(240, 190, 120), grip=(110, 116, 126), body=(150, 155, 165))
_cache = {}


def mesh(name):
    if name not in _cache:
        _cache[name] = to_mesh(C.load(name), 0.08)
    return _cache[name]


def placed(name, x_g, y_s, h_g, s=0.0, rot=False, rack=False):
    m = mesh(name).copy()
    if rack:
        return m
    if rot:
        m.apply_transform(np.diag([-1.0, -1.0, 1.0, 1.0]))
    m.apply_translation((0, 0, -s))
    T = np.array([[1, 0, 0, x_g], [0, -1, 0, y_s], [0, 0, -1, R.Z_SEAT + h_g - R.H_REST], [0, 0, 0, 1]], float)
    m.apply_transform(T)
    return m


def draw(ax, items):
    for m, col in items:
        n = m.face_normals
        c = np.clip(np.array(col)[None, :] / 255 * (0.35 + 0.65 * np.abs(n @ LIGHT))[:, None], 0, 1)
        ax.add_collection3d(Poly3DCollection(m.triangles, facecolors=c, edgecolors="none"))


def frame(ax, c, r, elev, azim, ortho=False):
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev, azim); ax.set_axis_off()
    if ortho:
        ax.set_proj_type("ortho")


def loaded():
    fig = plt.figure(figsize=(14, 7), dpi=90)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    items = []
    ysN, WN = R.layout(R.MODULES["N6"]); ysW, WW = R.layout(R.MODULES["W3"])
    items.append((mesh("Rack_Z_N6"), COL["rack"]))
    m = mesh("Rack_Z_W3").copy(); m.apply_translation((0, WN, 0)); items.append((m, COL["rack"]))
    for y, t in zip(ysN, ["claw", "scoop", "needle", "bitholder", "socket10", "flat"]):
        items.append((placed(f"Finger_Z_{t}", 0, y, R.H_REST), COL["finger"]))
    for y, t in zip(ysW, ["rake", "paddle", "wrench10"]):
        items.append((placed(f"Finger_Z_{t}", 0, WN + y, R.H_REST), COL["finger"]))
    draw(ax, items)
    frame(ax, (15, (WN + WW) / 2, -5), (WN + WW) / 2 * 0.42, 26, -122)
    ax.set_title("Z finger rack: Rack_Z_N6 (6 normal slots) + Rack_Z_W3 (3 wide slots), butted together",
                 fontsize=12, weight="bold")
    out = os.path.join(C.OUT, "rack_loaded.png")
    fig.tight_layout(); fig.savefig(out, facecolor="white"); print("wrote", out)


def sequence():
    ys, W = R.layout(R.MODULES["W3"])
    y = ys[1]
    steps = [("1. lower in front (gripper open)", -R.SLIDE, R.H_SLIDE + 25, 0.0, "carried"),
             ("2. down to 1.2 mm above the deck", -R.SLIDE, R.H_SLIDE, 0.0, "carried"),
             ("3. slide in", 0.0, R.H_SLIDE, 0.0, "carried"),
             ("4. set down, push 7.35 mm (releases)", 0.0, R.H_PUSH, R.PUSH, "seated"),
             ("5. lift clear: finger stays", 0.0, R.H_CLEAR, 0.0, "seated"),
             ("pick-up = the same moves backwards", 0.0, R.H_REST, 0.0, "carried")]
    fig = plt.figure(figsize=(18, 6.4), dpi=85)
    for k, (title, x_g, h_g, s, sv) in enumerate(steps):
        ax = fig.add_subplot(1, len(steps), k + 1, projection="3d", computed_zorder=False)
        cutaway = mesh("Rack_Z_W3").slice_plane([0, y, 0], [0, 1, 0], cap=True)     # section at the slot centre
        items = [(cutaway, COL["rack"])]
        items += [(placed("Receiver_Z", x_g, y, h_g), COL["grip"]), (placed("MHZL2-16D2_jaw", x_g, y, h_g), COL["grip"]),
                  (placed("MHZL2-16D2_body", x_g, y, h_g), COL["body"]),
                  (placed("Receiver_Z", x_g, y, h_g, rot=True), COL["grip"]),
                  (placed("MHZL2-16D2_jaw", x_g, y, h_g, rot=True), COL["grip"]),
                  (placed("Finger_Z_flat", x_g, y, h_g, rot=True), COL["other"])]
        if sv == "carried":
            items.append((placed("Finger_Z_wrench10", x_g, y, h_g), COL["finger"]))
        else:
            items.append((placed("Finger_Z_wrench10", 0, y, R.H_REST), COL["finger"]))
        draw(ax, items)
        frame(ax, (-2, y, 22), 58, 8, -90, ortho=True)
        ax.set_title(title, fontsize=10, weight="bold")
    fig.suptitle("Drop-off, rack cut at the slot centre (rack back to the right): served jaw = wrench10, other jaw = flat",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(C.OUT, "rack_sequence.png")
    fig.savefig(out, facecolor="white"); print("wrote", out)


if __name__ == "__main__":
    loaded()
    sequence()
