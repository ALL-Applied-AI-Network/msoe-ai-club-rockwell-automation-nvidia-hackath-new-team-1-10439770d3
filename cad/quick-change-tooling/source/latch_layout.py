"""latch_layout.py: every dimension of the push-push finger mount, for both insertion directions.

Frame = the official SMC STEP (mm), +X jaw at FULL OPEN, the same frame as build_fingers.py:
  Z tool axis (actuator front face Z 52.7, fingertips +Z), X jaw travel (gripping face X 6 open, X 0 closed),
  Y across. The -X jaw uses the same parts rotated 180 deg about Z.
SMC finger (exact, from the official model): X 13.45..18.45, Y +-4, Z 55..70, 2 x dia 3.4 holes along X at Z 59 / 66.

Parts per jaw: Receiver (bolted once to the SMC finger), Shuttle (the latch follower, captive in the receiver),
Finger (slides on, clicks on, clicks off; carries the heart track). Two variants:
  "Y": the finger slides on sideways (+Y for the +X jaw). Grip and table loads never act along the latch axis.
  "Z": the finger slides up along its own length. Pressing the fingertip on a surface with more than the eject spring
       force pushes it into the over-travel and releases it (see README).
Track coordinates (track_geom.py): u = shuttle direction, d = insertion depth. At the latched position the pin is at
the cusp (u_C, d_C).
"""
import math
import track_geom as TG

D = TG.derived()
U_C, D_C = D["cusp"]
D_MOUTH, D_V = D["d_mouth"], D["d_stop"]           # D_V: pin depth at the finger's hard stop
U0, U1 = D["u0"], D["u1"]                       # pin-centre travel limits of the shuttle (funnel mouth)
TRK = TG.track_polygon().bounds                 # physical cut: (u_min, d_min, u_max, d_max)
TRK_U0, TRK_D0, TRK_U1, TRK_D1 = TRK
TRK_UC = 0.5 * (TRK_U0 + TRK_U1)                # middle of the track across u

LATCH_STROKE = D_C - D_MOUTH                    # mouth -> latched
OVERTRAVEL = D_V - D_C                          # latched -> hard stop (the push to release / to latch)
STOP_MARGIN = 0.0                               # the hard stop is placed by track_geom (STOP_IN_WELL into the pockets)

# ---------------- SMC side (same as the verified Finger_base_adapter) ----------------
CLR = 0.15
FINGER = dict(x0=13.45, x1=18.45, y=4.0, z0=55.0, z1=70.0, holes=(59.0, 66.0))
SLEEVE_Z0, SLEEVE_Z1 = 55.5, 74.0
X_IN = 7.0                                      # receiver / finger-root inner face (1 mm behind the tip face)
X_OUT = FINGER["x1"] + CLR + 3.0                # 21.6 receiver outer face
Y_SL = FINGER["y"] + CLR + 2.45                 # 6.6 sleeve half-width
POCKET_TOP = FINGER["z1"] + 0.3                 # 70.3
M3_CLR, M3_NUT_AF, NUT_DEPTH = 3.4, 5.6, 3.0

# ---------------- finger ----------------
FACE_OPEN = 6.0                                 # tip gripping face X (12 mm gap open, 0 closed)
BEAM_X1, BEAM_Y = 18.0, 4.0
TIP_END = 52.7 + 76.0                           # 128.7: same faceplate-to-tip length (TCP) as the current blades
FINGER_IN = FACE_OPEN                           # finger inner face: flat over the whole length, so the finger prints
                                                # on it with the track facing up and no supports
PAD_LEN = 32.0                                  # soft-grip zone the simulation assumes (grip tape / TPU here)
ACC_HOLES = (106.0, 114.0)                      # generic M3 accessory / sensor-carrier holes along X

# ---------------- dovetail + track ----------------
X_I = 12.0                                      # interface plane: finger root X_IN..X_I, receiver X_I..X_OUT
TONGUE_H = 1.6                                  # tongue height (X)
DOVE = math.radians(45)                         # flank angle from the X axis: at 45 deg, pulling the tongue out
                                                # closes the clearance 1:1 (at 30 deg it took 1.73 x the clearance)
DOVE_CLR = 0.10                                 # dovetail clearance per side (flanks and floor); play_check.py
X_TF = X_I + TONGUE_H                           # 13.6 tongue face (carries the track)
X_GF = X_TF + DOVE_CLR                          # 13.7 groove floor (receiver)
WALL_TRK = 2.0                                  # material either side of the track inside the tongue neck (also the
                                                # dovetail's length across the track: more lever against rocking)
NECK_HW = 0.5 * (TRK_U1 - TRK_U0) + WALL_TRK    # tongue half-width at its neck (X_I)
FACE_HW = NECK_HW + TONGUE_H * math.tan(DOVE)   # tongue half-width at its face (X_TF)
G_NECK_HW = NECK_HW + DOVE_CLR                  # groove half-width at the mouth (X_I)
G_FLOOR_HW = G_NECK_HW + (X_GF - X_I) * math.tan(DOVE)
TRACK_DEPTH = 2.15                              # into the tongue face
X_TRK = X_TF - TRACK_DEPTH                      # 11.45 track floor

# ---------------- shuttle (printed follower) and its T-slot ----------------
PIN_D = TG.PIN_D
PIN_TIP_X = X_TRK + 0.3                         # peg end, 0.3 above the track floor
LIP_T = 1.0                                     # T-slot lip thickness (X_GF .. X_GF + LIP_T)
NECK_W = PIN_D + 0.3                            # slot the peg runs in (width along d)
HEAD_T, HEAD_W, HEAD_L = 1.3, 4.6, 6.0          # shuttle head: thickness X, width along d, length along u. Long
                                                # along u: the side push on the peg tip tries to tilt the head in its
                                                # slot, and a short head turns that into binding (6.0: ~0.5 x the push
                                                # on each end instead of 0.85 x at 3.6)
CHAN_T, CHAN_W = HEAD_T + 0.2, HEAD_W + 0.4     # head channel
X_LIP = X_GF + LIP_T                            # 14.75
X_CHAN = X_LIP + CHAN_T                         # 16.25 channel bottom
HEAD_X0 = X_LIP + 0.1                           # shuttle head sits 0.1 above the lips at rest
PEG_LEN = HEAD_X0 - PIN_TIP_X
SH_U0, SH_U1 = U0, U1                           # pin-centre range the T-slot allows = the funnel mouth exactly
M2_PILOT = 1.8                                  # self-tapping M2 retaining screw across the channel's open end

# ---------------- eject spring ----------------
SPRING = dict(od=3.0, wire=0.3, free=20.0, rate=0.7)    # N/mm, a common assortment size; see README
SPRING_BORE_D, SPRING_BORE_DEPTH, SPRING_SEAT_D = 3.4, 7.0, 2.9   # last 1 mm undersize: the last coil presses in

# ======================================================================== variant Y: slides on along +Y
class VarY:
    name = "Y"
    ins = (0.0, 1.0, 0.0)                       # finger moves this way to latch
    ROOT_Z0 = SLEEVE_Z1 + 0.3                   # 74.3, under the sleeve
    ZC = ROOT_Z0 + 0.4 + FACE_HW                # tongue centre height: wide face just below the root top
    Z_B = ZC + G_FLOOR_HW + 1.5                 # lap bottom (shoulder plane)
    ROOT_Z1 = Z_B + 0.05                        # root / beam junction: the beam shoulder sits 0.05 under the lap
    # along Y (latched): leading face Y_LE (track mouth), end wall face Y_EW, root trailing end Y_TR
    END_WALL_T = SPRING_BORE_DEPTH + 1.5
    HALF_SPAN = 14.9                            # receiver overall +-Y
    Y_EW0 = HALF_SPAN - END_WALL_T              # end-wall inner face
    Y_LE = Y_EW0 - OVERTRAVEL - STOP_MARGIN     # finger leading face at latched
    Y_P = Y_LE - LATCH_STROKE                   # pin (shuttle) Y
    TRK_TAIL = D_C - TRK_D1                     # (negative) track bottom relative to the pin, along +Y on the finger
    Y_TR = Y_P + TRK_TAIL - 1.6                 # root trailing end
    Z_P = None                                  # set below: pin height
    SPRING_AT = (9.5, None)                     # spring axis X, Z (Z set below)

    @classmethod
    def track_to_world(cls, u, d, s=0.0):
        """track point (u, d) on the finger, finger shifted by s along the insertion direction from latched"""
        return (X_TF, cls.Y_P + (D_C - d) + s, cls.Z_P + (u - U_C))

    @classmethod
    def shuttle_at(cls, u):
        """shuttle reference (peg axis at the groove floor) when the pin centre is at lateral u"""
        return (X_GF, cls.Y_P, cls.Z_P + (u - U_C))


VarY.Z_P = VarY.ZC + (U_C - TRK_UC)             # centre the track across the tongue
VarY.SPRING_AT = (9.5, VarY.ZC)


# ======================================================================== variant Z: slides up along -Z
class VarZ:
    name = "Z"
    ins = (0.0, 0.0, -1.0)
    STOP_Z = SLEEVE_Z1                           # root top hits the receiver here at the bottom of the push
    Z_LE = STOP_Z + OVERTRAVEL + STOP_MARGIN     # root top at latched (track mouth)
    Z_P = Z_LE + LATCH_STROKE                    # pin height
    ROOT_Z1 = Z_P + (TRK_D1 - D_C) + 1.6         # root bottom (track tail + 1.6)
    UPPER_Z0 = 64.0                              # the receiver widens above the root to hold the spring
    HALF_W = FACE_HW + DOVE_CLR + 1.7            # receiver lap +-Y
    ROOT_HW = HALF_W - 0.4                       # finger root +-Y
    Z_B = ROOT_Z1 + 0.1                          # lap bottom
    SHOULDER_GAP = OVERTRAVEL + STOP_MARGIN + 0.3   # beam shoulder this far below the lap at latched
    Z_SH = Z_B + SHOULDER_GAP
    Y_C = 0.0                                    # tongue centre
    Y_P = None
    SPRING_AT = (9.5, 6.0)                       # spring axis X, Y (clear of the nut trap and cross bolt)

    @classmethod
    def track_to_world(cls, u, d, s=0.0):
        return (X_TF, cls.Y_P + (u - U_C), cls.Z_P + (d - D_C) - s)

    @classmethod
    def shuttle_at(cls, u):
        return (X_GF, cls.Y_P + (u - U_C), cls.Z_P)


VarZ.Y_P = VarZ.Y_C + (U_C - TRK_UC)

VARIANTS = {"Y": VarY, "Z": VarZ}


def report():
    print(f"track: physical u {TRK_U0:.2f}..{TRK_U1:.2f} ({TRK_U1 - TRK_U0:.2f}), d {TRK_D0:.2f}..{TRK_D1:.2f} "
          f"({TRK_D1 - TRK_D0:.2f}); latch stroke {LATCH_STROKE:.2f}, over-travel (click) {OVERTRAVEL:.2f}")
    print(f"tongue: neck half-width {NECK_HW:.3f}, face half-width {FACE_HW:.3f}; groove {G_NECK_HW:.3f} / {G_FLOOR_HW:.3f}")
    print(f"shuttle peg length {PEG_LEN:.2f}, T-slot pin range u {SH_U0:.2f}..{SH_U1:.2f}")
    v = VarY
    print(f"Y: ZC {v.ZC:.3f} Z_P {v.Z_P:.3f} Z_B {v.Z_B:.3f}; Y_TR {v.Y_TR:.3f} Y_P {v.Y_P:.3f} Y_LE {v.Y_LE:.3f} "
          f"end wall {v.Y_EW0:.3f}..{v.HALF_SPAN}; root length {v.Y_LE - v.Y_TR:.2f}")
    v = VarZ
    print(f"Z: root Z {v.Z_LE:.3f}..{v.ROOT_Z1:.3f}, Z_P {v.Z_P:.3f}, lap bottom {v.Z_B:.3f}, shoulder {v.Z_SH:.3f}; "
          f"Y_P {v.Y_P:.3f}, lap +-{v.HALF_W:.3f}")
    s = SPRING
    for nm, L_lat, L_bot in (("Y", SPRING_BORE_DEPTH + (VarY.Y_EW0 - VarY.Y_LE), SPRING_BORE_DEPTH + STOP_MARGIN),
                             ("Z", SPRING_BORE_DEPTH + (VarZ.Z_LE - VarZ.STOP_Z), SPRING_BORE_DEPTH + 0.0)):
        print(f"spring ({nm}): length latched {L_lat:.2f} -> {s['rate'] * (s['free'] - L_lat):.1f} N, "
              f"bottom {L_bot:.2f} -> {s['rate'] * (s['free'] - L_bot):.1f} N (solid ~{s['wire'] * (s['free'] / 1.5 + 2):.1f})")


if __name__ == "__main__":
    report()
