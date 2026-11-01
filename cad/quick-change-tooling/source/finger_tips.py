"""finger_tips.py: the 20 interchangeable tips for the push-push fingers (variants Y and Z).

Every tip is built by build_latch_sw.py on top of the standard finger: same root, dovetail and heart track, same beam
(X 6..18, Y +-4, ending at Z 128.7), so the latch is exactly the proven one. Tips only add or remove material at
Z >= 120 (below the Z variant's accessory hole), in the frame of the +X jaw at full open:
  X 6 = gripping face (the print bed: fingers print face down, track up), X grows away from the other finger,
  Y across, Z along the finger toward the tip.
Rules every tip keeps (checked by check_tips.py):
  - nothing below X 6: the tips print without supports, and two fingers can always close fully together (faces meet
    in the plane X 0, like the flat fingers);
  - overhangs in print orientation (X up) are 45 deg or less, or short bridges.
Tool tips (bits, drivers, sockets, ...) sit on an axis parallel to Z; its X is in AXIS_X (open position; subtract
6 when the gripper is fully closed). Set the robot's TCP there.
"""
import math

TE = 128.7                      # standard tip end (Z)
Z0 = 120.0                      # tip features start here


def _deps():
    import build_latch_sw as B
    return B


# ---------------------------------------------------------------- primitives (world coordinates, mm)
def block(p, x0, x1, y0, y1, z0, z1, label):
    B = _deps()
    B.sk(p, "Z", z0); B.rect(p, (x0, y0, z0), (x1, y1, z0))
    B.boss(p, z1 - z0, label, "+Z")


def prof_y(p, pts_xz, y0, y1, label, cut=False):
    """profile in the XZ plane, extruded from Y = y0 to y1"""
    B = _deps()
    B.sk(p, "Y", y0); p.poly([(x, y0, z) for x, z in pts_xz])
    if cut:
        B.cut(p, y1 - y0, label, "+Y")
    else:
        B.boss(p, y1 - y0, label, "+Y")


def prof_x(p, pts_yz, x0, x1, label, cut=False):
    """profile in the YZ plane, extruded from X = x0 to x1"""
    B = _deps()
    B.sk(p, "X", x0); p.poly([(x0, y, z) for y, z in pts_yz])
    if cut:
        B.cut(p, x1 - x0, label, "+X")
    else:
        B.boss(p, x1 - x0, label, "+X")


def prof_z(p, pts_xy, z0, z1, label, cut=False):
    """profile in the XY plane, extruded from Z = z0 toward z1 (either way)"""
    B = _deps()
    d = "+Z" if z1 > z0 else "-Z"
    B.sk(p, "Z", z0); p.poly([(x, y, z0) for x, y in pts_xy])
    if cut:
        B.cut(p, abs(z1 - z0), label, d)
    else:
        B.boss(p, abs(z1 - z0), label, d)


def circle_z(p, c, r, z0, z1, label, cut=False):
    B = _deps()
    d = "+Z" if z1 > z0 else "-Z"
    B.sk(p, "Z", z0); p.circle((c[0], c[1], z0), r)
    if cut:
        B.cut(p, abs(z1 - z0), label, d)
    else:
        B.boss(p, abs(z1 - z0), label, d)


def hexagon(cx, cy, af, flats_normal_to="X", grow=0.0):
    """hexagon (XY) across flats af; flats normal to X = a flat on top and bottom in print (short bridge)"""
    r = (af + 2 * grow) / math.sqrt(3)
    a0 = 30 if flats_normal_to == "X" else 0           # flats normal to X: vertices at +-Y (angles 30, 90, ...)
    return [(cx + r * math.cos(math.radians(a0 + 60 * k)), cy + r * math.sin(math.radians(a0 + 60 * k))) for k in range(6)]


def star(cx, cy, r_out, r_in, n=6, a0=180.0):
    pts = []
    for k in range(2 * n):
        r = r_out if k % 2 == 0 else r_in
        a = math.radians(a0 + 180.0 / n * k)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def collar(p, half_y=5.0, z1=132.0):
    block(p, 6.0, 18.0, -half_y, half_y, Z0, z1, "Tool collar")


# ---------------------------------------------------------------- gripping / bulk tips
def claw(p):
    block(p, 6.0, 11.0, -11.5, 11.5, Z0, 130.0, "Claw base")
    tine = [(6, 128), (11, 128), (19, 136), (19, 141), (10, 150), (6, 150), (6, 147), (12.5, 140.5), (12.5, 135.5),
            (6, 129)]
    for k, yc in enumerate((-9.5, -3.17, 3.17, 9.5)):
        prof_y(p, tine, yc - 2.0, yc + 2.0, f"Claw tine {k + 1}")


def scoop(p):
    block(p, 6.0, 16.0, -12.5, 12.5, Z0, 150.0, "Scoop body")
    prof_z(p, [(5.0, -11.0), (6.0, -10.5), (12.0, -4.5), (12.0, 4.5), (6.0, 10.5), (5.0, 11.0)], 125.0, 147.5,
           "Scoop trough (45 deg walls, prints without supports)", cut=True)


def rake(p):
    block(p, 6.0, 10.0, -20.0, 20.0, 124.0, 131.0, "Rake bar")
    block(p, 10.0, 14.0, -6.0, 6.0, Z0, 131.0, "Rake web")
    B = _deps()
    B.sk(p, "Z", 131.0)
    for y in (-18.0, -12.0, -6.0, 0.0, 6.0, 12.0, 18.0):
        B.rect(p, (6.0, y - 1.5, 131.0), (9.5, y + 1.5, 131.0))
    B.boss(p, 15.0, "Rake teeth", "+Z")


def paddle(p):
    block(p, 6.0, 9.5, -20.0, 20.0, Z0, 145.0, "Paddle plate")
    block(p, 9.5, 12.5, -20.0, 20.0, Z0, 124.0, "Paddle stiffening rib")
    B = _deps()
    B.sk(p, "Y", -20.0)
    for z in [123.0 + 3.0 * k for k in range(8)]:
        B.rect(p, (5.0, -20.0, z), (6.6, -20.0, z + 1.2))
    B.cut(p, 40.0, "Paddle grip grooves", "+Y")


def needle(p):
    prof_y(p, [(6, Z0), (18, Z0), (18, TE), (9, 158), (6, 158)], -3.0, 3.0, "Needle nose")
    prof_x(p, [(3.01, 132.0), (3.01, 158.5), (1.0, 158.5)], 5.0, 20.0, "Needle taper +Y", cut=True)
    prof_x(p, [(-3.01, 132.0), (-1.0, 158.5), (-3.01, 158.5)], 5.0, 20.0, "Needle taper -Y", cut=True)
    B = _deps()
    B.sk(p, "Y", -5.0)
    for z in [145.0 + 1.6 * k for k in range(7)]:
        B.rect(p, (5.0, -5.0, z), (6.4, -5.0, z + 0.7))
    B.cut(p, 10.0, "Needle serrations", "+Y")


def vcradle(p):
    block(p, 6.0, 18.0, -16.0, 16.0, Z0, 142.0, "Cradle pad")
    prof_z(p, [(5.0, -9.0), (6.0, -9.0), (14.0, -1.0), (14.0, 1.0), (6.0, 9.0), (5.0, 9.0)], Z0, 143.0,
           "90 deg V cradle (vertical cylinders)", cut=True)


def spatula(p):
    prof_y(p, [(6, Z0), (14, Z0), (14, TE), (7.2, 152), (6, 152)], -11.0, 11.0, "Spatula blade")


def hook(p):
    prof_x(p, [(-3.0, Z0), (3.0, Z0), (3.0, 146.0), (9.0, 146.0), (9.0, 138.0), (15.0, 138.0), (15.0, 152.0),
               (-3.0, 152.0)], 6.0, 12.0, "Side hook (on the print bed)")


def magnet(p):
    block(p, 6.0, 18.0, -7.0, 7.0, Z0, 131.5, "Magnet pad")
    circle_z(p, (12.0, 0.0), 5.1, 131.5, 128.3, "Magnet pocket (10 x 3 mm disc)", cut=True)


# ---------------------------------------------------------------- tool tips (axis along Z)
def bit_holder(p):
    block(p, 6.0, 18.0, -6.0, 6.0, Z0, 150.0, "Bit holder body")
    prof_z(p, hexagon(12.0, 0.0, 6.45), 150.0, 137.0, "1/4 in hex socket", cut=True)
    circle_z(p, (12.0, 0.0), 3.1, 150.0, 134.0, "Magnet pocket (6 x 3 mm)", cut=True)


def flat_sd(p):
    collar(p)
    prof_z(p, hexagon(9.0, 0.0, 6.0), 132.0, 150.0, "Shaft")
    prof_x(p, [(-3.0, 150.0), (3.0, 150.0), (0.5, 158.0), (-0.5, 158.0)], 6.0, 12.0, "Flat blade 6 mm")


def ph2(p):
    collar(p)
    prof_z(p, hexagon(9.0, 0.0, 6.0), 132.0, 150.0, "Shaft")
    c = (9.0, 0.0); w, h = 0.8, 3.0
    u = (math.cos(math.radians(45)), math.sin(math.radians(45))); v = (-u[1], u[0])
    def P(a, b):
        return (c[0] + a * u[0] + b * v[0], c[1] + a * u[1] + b * v[1])
    cross = [P(h, -w), P(h, w), P(w, w), P(w, h), P(-w, h), P(-w, w), P(-h, w), P(-h, -w), P(-w, -w), P(-w, -h),
             P(w, -h), P(w, -w)]
    B = _deps()
    B.sk(p, "Z", 150.0); p.poly([(x, y, 150.0) for x, y in cross])
    p.boss_draft(p.done(), 5.0, 5.0, False, "PH2-style cross (printed, light duty)")


def hex_key(af):
    def f(p):
        collar(p)
        prof_z(p, hexagon(6.0 + (af - 0.05) / 2, 0.0, af - 0.05), 132.0, 146.0, f"Hex key {af:g} mm")
    return f


def torx25(p):
    collar(p)
    prof_z(p, star(6.0 + 2.2, 0.0, 2.2, 1.6), 132.0, 142.0, "T25-style star (printed, light duty)")


def socket_w(af):
    return math.ceil(af / math.cos(math.radians(30)) + 4.4)


def socket(af, bolt):
    def f(p):
        W = socket_w(af)
        ax = 6.0 + W / 2
        top = TE + 14.0
        block(p, 6.0, 6.0 + W, -W / 2, W / 2, Z0, top, f"Socket {af:g} body")
        prof_z(p, hexagon(ax, 0.0, af + 0.25), top, top - 7.0, f"Hex {af:g} mm", cut=True)
        circle_z(p, (ax, 0.0), bolt / 2, top, top - 16.0, f"Bolt clearance {bolt:g}", cut=True)
    return f


WRENCH_C = (14.0, 13.0)          # nut centre (open position): beside the beam, so a bolt can pass through the nut


def wrench10(p):
    block(p, 6.0, 25.5, -4.0, 24.0, TE - 3.0, TE + 3.0, "Wrench head")
    r = 10.2 / math.sqrt(3); cx, cy = WRENCH_C
    prof_z(p, [(5.0, cy - 5.1), (cx + r / 2, cy - 5.1), (cx + r, cy), (cx + r / 2, cy + 5.1), (5.0, cy + 5.1)],
           TE - 3.5, TE + 3.5, "Open jaw 10 mm (opens toward the other finger)", cut=True)


def probe(p):
    collar(p)
    circle_z(p, (10.0, 0.0), 4.0, 132.0, 152.0, "Probe shaft")
    B = _deps()
    B.sk(p, "Z", 152.0); p.circle((10.0, 0.0, 152.0), 4.0)
    p.boss_draft(p.done(), 3.0, 45.0, False, "Probe point")


# ---------------------------------------------------------------- catalogue
# name: (builder, what it's for, pair it with, best variant, tool axis X (open) or None, print note)
TIPS = {
    "claw":      (claw, "4-tine claw: two of them close into a cage for bulk loose parts, bags, odd shapes",
                  "claw", "Y or Z", None, "no supports; tines rise at 45 deg"),
    "scoop":     (scoop, "clamshell half: two close into a closed bucket (granules, screws, beads)",
                  "scoop", "Y or Z", None, "no supports; 45 deg trough walls, 9 mm bridge on top"),
    "rake":      (rake, "7-tooth rake: sweep, spread or drag loose parts", "rake or flat", "Y or Z", None,
                  "no supports"),
    "paddle":    (paddle, "40 mm wide grooved paddle: boxes, books, big flat objects", "paddle", "Y or Z", None,
                  "no supports"),
    "needle":    (needle, "needle nose with serrations: small screws, pins, wires", "needle", "Y or Z", None,
                  "no supports"),
    "vcradle":   (vcradle, "90 deg V cradle: upright cylinders 12-40 mm (bottles, tubes, pens)", "vcradle",
                  "Y or Z", None, "no supports; V walls at 45 deg"),
    "spatula":   (spatula, "thin spatula blade: slide under flat parts, cards, paper", "flat", "Y", None,
                  "no supports"),
    "hook":      (hook, "side hook: lift by rings, handles, loops; pull drawers", "flat", "Y", None,
                  "no supports"),
    "magnet":    (magnet, "pocket for a 10 x 3 mm neodymium disc: screws, washers, nuts, bulk steel parts",
                  "flat or magnet", "Y or Z", None, "no supports; glue the magnet in"),
    "bitholder": (bit_holder, "1/4 in hex bit holder with magnet: takes ANY steel bit (Phillips, Torx, hex, ...)",
                  "flat", "Y", 12.0, "no supports; push-fit the bit, 6 x 3 mm magnet at the bottom"),
    "flatsd":    (flat_sd, "flat screwdriver, 6 mm blade (printed)", "flat", "Y", 9.0, "no supports"),
    "ph2":       (ph2, "Phillips PH2-style screwdriver (printed)", "flat", "Y", 9.0, "no supports"),
    "hex3":      (hex_key(3.0), "3 mm hex key (printed)", "flat", "Y", 7.5, "no supports"),
    "hex5":      (hex_key(5.0), "5 mm hex key (printed)", "flat", "Y", 8.5, "no supports"),
    "torx25":    (torx25, "T25-style star driver (printed)", "flat", "Y", 8.2, "no supports"),
    "socket7":   (socket(7.0, 4.5), "7 mm socket (M4 nuts)", "flat", "Y", 6.0 + socket_w(7.0) / 2, "no supports"),
    "socket8":   (socket(8.0, 5.5), "8 mm socket (M5 nuts)", "flat", "Y", 6.0 + socket_w(8.0) / 2, "no supports"),
    "socket10":  (socket(10.0, 6.6), "10 mm socket (M6 nuts and bolt heads)", "flat", "Y", 6.0 + socket_w(10.0) / 2,
                  "no supports"),
    "wrench10":  (wrench10, "10 mm open-end wrench; the jaw opens toward the other finger, nut at Y +13", "flat", "Y",
                  WRENCH_C[0], "no supports"),
    "probe":     (probe, "pointed probe: press buttons, poke, push parts into place", "flat", "Y", 10.0,
                  "no supports"),
}
