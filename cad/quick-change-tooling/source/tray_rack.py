"""tray_rack.py: parking rack for the Z-variant fingers (tool changer). Builds out/Rack_Z_<module>.SLDPRT/STEP/STL.

    python tray_rack.py [module ...]        (default: every module in MODULES)

How it works: the fingers hang in a row of forks, track end up. Every Z finger rests on the same two faces, the
undersides of the root "wings" (finger X 6..12, |Y| 4..11.8, Z 115.45); nothing of the receiver ever reaches them.
A plain drop-in pocket cannot work for every tip: rake, paddle, cradle and wrench reach right under the wings. So the
finger is lowered in front of the rack, slid in sideways (tips pass under the 2.5 mm deck), set down, and only then
does the latch work, straight down as always:
  drop-off: lower in front, slide in, set down, push (the latch releases), lift away;
  pick-up:  come down onto the parked root, push (the latch catches), lift 1.2 mm, slide out, lift away.
The gripper stays OPEN, and the jaw being served faces the back of the rack. To serve the other jaw, turn the wrist
180 deg; its finger is identical, so it parks in the same kind of slot. The other jaw's finger hangs in front of the
rack meanwhile (open air: nothing of the rack lies in front of x 3).

Rack frame (mm): x = finger X of the served jaw (gripper open; its gripping face at x 6, gripper axis at x 0 when the
finger is parked), y along the rack, h up from the deck top (= the wing plane). Gripper frame -> rack frame:
x = X + x_g, y = y_slot - Y, h = Z_SEAT - Z + (h_g - H_REST) for the gripper at (x_g, y_slot, h_g), where h_g is the
height of the actuator front face centre (SMC frame Z 52.7) above the deck.
"""
import math, os, sys
sys.path.insert(0, os.environ.get("ATEAM_CAD_HELPERS", os.path.dirname(os.path.abspath(__file__))))
import latch_layout as L

Z_SEAT = L.VarZ.Z_SH                    # 115.448: wing undersides (finger frame) = deck top
H_REST = Z_SEAT - 52.7                  # 62.75: actuator face above the deck, finger latched and seated
TIP_Z0 = 120.0                          # every tip starts below this (finger_tips.Z0); only the beam is above it

DECK_T = 2.5                            # deck thickness: tips clear its underside by 2.05 parked, 0.85 sliding
LIFT = 1.2                              # wings this far above the deck while sliding
LIP_H, LIP_X0, LIP_X1 = 0.5, 5.0, 5.8   # stop lip in front of the root (root face x 6): parked fingers can't creep out;
                                        # the receiver starts at x 7, so the lip never meets it
SLOT_HW, SLOT_END = 4.2, 18.3           # beam slot (beam |Y| 4, X 6..18): 0.2 a side, 0.3 behind the beam end
MOUTH_X, MOUTH_HW, MOUTH_LEN = 3.5, 6.2, 3.5   # deck front and the slot's lead-in flare
SLIDE = 24.0                            # slide length: the widest tip (wrench, x 25.5) is 2 mm in front of the deck
X_FRONT, X_BACK0, X_BACK1 = 3.0, 27.0, 30.0     # ribs start at x 3; back wall x 27..30 (wrench reaches 25.5)
H_TOP = 15.0                            # ribs and back wall rise to here above the deck
H_FLOOR, FLOOR_T = -54.0, 3.0           # rack stands on its floor; deepest tip parked at -42.6 (needle, flatsd)
RIB_IN = 13.5                           # ribs and stiffeners keep |y - y_slot| >= 13.5: receiver +-12.19, root +-11.79

# slot classes: half-width of the slot (pitch = sum of neighbouring halves) and where the full-height rib starts.
# N: tips within |Y| 12.5; W: rake, paddle, cradle, hook, wrench (|Y| up to 24). W slots also get above-deck
# stiffeners at RIB_IN (the deck alone would span 21 mm there).
CLASSES = {"N": dict(half=16.0, rib=RIB_IN), "W": dict(half=27.5, rib=25.0)}
WIDE = ("rake", "paddle", "vcradle", "hook", "wrench10")
MODULES = {"N6": "NNNNNN", "W3": "WWW"}          # 192 x 27 x 69 and 165 x 27 x 69: each fits a 220 mm bed

# robot waypoints: actuator face centre (x_g, h_g) (x 0 = gripper axis with the finger parked, h 0 = deck top)
PUSH = L.OVERTRAVEL - 0.4               # 7.35: push 0.4 short of the hard stop (the track works to 0.8 short)
H_SLIDE = H_REST + LIFT                 # 63.95
H_PUSH = H_REST - PUSH                  # 55.40
H_CLEAR = H_REST + (L.VarZ.Z_SH - L.VarZ.Z_LE) - (L.VarZ.Z_SH - L.VarZ.Z_B) + 4.0   # receiver 4 mm above the roots
WAYPOINTS = {
    "drop": [("above, in front", -SLIDE, None), ("down in front", -SLIDE, H_SLIDE), ("slide in", 0.0, H_SLIDE),
             ("set down", 0.0, H_REST), ("push (releases)", 0.0, H_PUSH), ("lift clear", 0.0, H_CLEAR)],
    "pick": [("above the root", 0.0, H_CLEAR), ("push (catches)", 0.0, H_PUSH), ("lift 1.2", 0.0, H_SLIDE),
             ("slide out", -SLIDE, H_SLIDE), ("lift away", -SLIDE, None)],
}


def slot_class(tip):
    return "W" if tip in WIDE else "N"


def layout(classes):
    """slot centres y and module width for a string of classes"""
    ys, y = [], 0.0
    for c in classes:
        y += CLASSES[c]["half"]; ys.append(y); y += CLASSES[c]["half"]
    return ys, y


def ribs(classes):
    """full-height rib y-ranges: one at each module end and one between neighbouring slots"""
    ys, W = layout(classes)
    out, y0 = [], 0.0
    for k, c in enumerate(classes):
        h, r = CLASSES[c]["half"], CLASSES[c]["rib"]
        lo, hi = ys[k] - h, ys[k] + h
        if k == 0:
            out.append([lo, lo + (h - r)])
        else:
            out[-1][1] = lo + (h - r)
        out.append([hi - (h - r), hi])
    return [tuple(r) for r in out]


def slot_poly(ys):
    return [(MOUTH_X - 1.0, ys - MOUTH_HW - 1.0), (MOUTH_X, ys - MOUTH_HW), (MOUTH_X + MOUTH_LEN, ys - SLOT_HW),
            (SLOT_END, ys - SLOT_HW), (SLOT_END, ys + SLOT_HW), (MOUTH_X + MOUTH_LEN, ys + SLOT_HW),
            (MOUTH_X, ys + MOUTH_HW), (MOUTH_X - 1.0, ys + MOUTH_HW + 1.0)]


# ======================================================================== SolidWorks
def build(name):
    import build_latch_sw as B
    from swkit2 import GPart
    classes = MODULES[name]
    ys, W = layout(classes)
    p = GPart(f"Rack_Z_{name}")
    B.sk(p, "Z", H_FLOOR); B.rect(p, (X_FRONT, 0.0, H_FLOOR), (X_BACK1, W, H_FLOOR))
    B.boss(p, FLOOR_T, "Floor", "+Z")
    h0 = H_FLOOR + FLOOR_T
    B.sk(p, "Z", h0); B.rect(p, (X_BACK0, 0.0, h0), (X_BACK1, W, h0))
    B.boss(p, H_TOP - h0, "Back wall", "+Z")
    B.sk(p, "Z", h0)
    for a, b in ribs(classes):
        B.rect(p, (X_FRONT, a, h0), (X_BACK0, b, h0))
    B.boss(p, H_TOP - h0, "Ribs", "+Z")
    B.sk(p, "Z", -DECK_T); B.rect(p, (MOUTH_X, 0.0, -DECK_T), (X_BACK0, W, -DECK_T))
    B.boss(p, DECK_T, "Deck", "+Z")
    B.sk(p, "Z", 0.0); B.rect(p, (LIP_X0, 0.0, 0.0), (LIP_X1, W, 0.0))
    B.boss(p, LIP_H, "Stop lip", "+Z")
    stiff = [(y + s * RIB_IN, y + s * (RIB_IN + 2.0)) for y, c in zip(ys, classes) if c == "W" for s in (-1, 1)]
    if stiff:
        B.sk(p, "Z", LIP_H)
        for a, b in stiff:
            B.rect(p, (X_FRONT, min(a, b), LIP_H), (X_BACK0, max(a, b), LIP_H))
        B.boss(p, H_TOP - LIP_H, "Deck stiffeners", "+Z")
    B.sk(p, "Z", LIP_H)
    for y in ys:
        p.poly([(x, yy, LIP_H) for x, yy in slot_poly(y)])
    B.cut(p, LIP_H + DECK_T, "Beam slots", "-Z")
    B.sk(p, "Z", H_FLOOR)
    for y in (ys[0], ys[-1]):
        p.circle((16.5, y, H_FLOOR), 1.7)
    B.cut(p, FLOOR_T, "M3 mounting holes", "+Z")
    B.save(p, (60, 140, 200))


def report():
    for n, cl in MODULES.items():
        ys, W = layout(cl)
        print(f"Rack_Z_{n}: {W:.1f} x {X_BACK1 - X_FRONT:.1f} x {H_TOP - H_FLOOR:.1f} mm, slots at y "
              + ", ".join(f"{y:.1f}" for y in ys) + f"; ribs {[(round(a, 1), round(b, 1)) for a, b in ribs(cl)]}")
    print(f"heights (actuator face above deck): rest {H_REST:.2f}, slide {H_SLIDE:.2f}, push {H_PUSH:.2f}, "
          f"clear {H_CLEAR:.2f}")


if __name__ == "__main__":
    report()
    if "--report" not in sys.argv:
        from swkit2 import g, app
        g(app(), "CloseAllDocuments", True)
        for n in [a for a in sys.argv[1:] if a in MODULES] or list(MODULES):
            build(n)
