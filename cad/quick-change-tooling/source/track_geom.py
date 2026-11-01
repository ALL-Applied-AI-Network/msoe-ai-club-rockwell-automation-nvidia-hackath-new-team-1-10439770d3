"""track_geom.py: the push-push heart track.

Track-plane coordinates: d = depth along the insertion direction (the pin moves to LARGER d when the finger is pushed
in, to smaller d when the eject spring pushes it out); u = lateral, the direction the pin's shuttle slides freely.

Two kinds of geometry, kept separate so every wall is exact:
  - ISLAND: the physical island (a polygon) that the pin steers around.
  - OUTER:  the outer limit of the pin CENTRE (configuration space): funnel, throat, outer walls, floor ramps.
The pin centre may go anywhere in FREE = OUTER minus (ISLAND grown by RP), RP = pin radius + clearance.
The physical cut is (OUTER grown by RP) minus ISLAND: its outer wall is OUTER offset by RP (lines plus arcs at
OUTER's convex corners, built exactly by `outer_outline()`), its island is ISLAND itself.

Cycle (checked in simulate.py):
  push 1    : through the throat onto the island's left flank, down the left channel; the floor ramp moves it right
              into valley V1, under the island's left underside.
  release 1 : the eject spring lifts it into the underside, which steers it up into the cusp: LATCHED.
  push 2    : straight down from the cusp onto ramp P-V2, which carries it right, past the island's lower-right
              corner R, into valley V2.
  release 2 : up the island's lower-right edge R-B (steers right), up the right channel, the top wall funnels it
              left over the island tip, through the throat and out of the mouth: RELEASED.
Every steering wall is at most 50 deg off the motion axis, so it slides with a total friction coefficient
(track + shuttle slot) up to 1/tan(50 deg) = 0.84.
"""
import math

# ---------------- design parameters (mm) ----------------
PIN_D = 2.0                    # printed peg on the shuttle (a 2 mm steel pin fits the same track)
CLR = 0.15                     # radial clearance, pin to wall
RP = PIN_D / 2 + CLR
MAX_WALL = 50.0                # deg off the motion axis for the spring-driven steering walls
T50 = math.tan(math.radians(MAX_WALL))
FUNNEL_WALL = 45.0             # deg off the insertion axis for the entry funnel (pushed by hand)
FLOOR_WALL = 40.0              # deg off the motion axis for the floor ramps (the moves made while PUSHING, which
                               # carry the shuttle sideways): works up to a total friction of 1/tan(40) = 1.19
RHO_I = 0.3                    # fillet on every convex island corner (smooth, and it is what a nozzle prints anyway)
RHO_O = 0.3                    # fillet on every outer-wall corner that sticks into the track
RI = RP + RHO_I                # pin-centre radius around the island's fillet centres
RO = RP + RHO_O

# physical island (u, d), before the fillets: tip T, left flank to K, left side down to L, underside up to the notch
# C, steep notch wall down to R, up to B, straight up to B2. Built from a few numbers so the checks can steer the shape.
UNDER_LEN = 7.6                # C -> L along the underside (MAX_WALL off d): long, so pocket V1 sits far left of the cusp
NOTCH_WALL, NOTCH_LEN = 12.0, 2.6   # C -> R: deg off d, length. Deep notch: the latched pin must sink to escape
RB_LEN = 1.6                   # R -> B, up-right at MAX_WALL
LEFT_H = 1.6                   # K -> L, the vertical left side
TIP_U = 0.3                    # u of the tip T; T -> K runs at 45 deg
RIGHT_H = 1.5                  # B -> B2, the vertical right side (keeps the neck beside the notch thick)


def _island():
    s50, c50 = math.sin(math.radians(MAX_WALL)), math.cos(math.radians(MAX_WALL))
    C = (0.0, 0.0)
    L = (-UNDER_LEN * s50, UNDER_LEN * c50)
    R = (NOTCH_LEN * math.sin(math.radians(NOTCH_WALL)), NOTCH_LEN * math.cos(math.radians(NOTCH_WALL)))
    B = (R[0] + RB_LEN * s50, R[1] - RB_LEN * c50)
    K = (L[0], L[1] - LEFT_H)
    T = (TIP_U, K[1] - (TIP_U - K[0]))
    B2 = (B[0], B[1] - RIGHT_H)
    return [T, K, L, C, R, B, B2]


ISLAND = _island()
T_, K_, L_, C_, R_, B_, B2_ = ISLAND

GAP = 0.5                      # free play of the pin centre in the channels
WELL_W, WELL_D = 0.3, 1.8      # straight pocket under each valley (pin-centre width, depth): the sideways move is
                               # finished at the top of the pocket, the rest of the stroke is straight
STOP_IN_WELL = 0.9             # the finger's hard stop is this far down the pocket: the stroke may be 0.9 mm short
                               # or long (print error) and the pin still ends up in the right place
M_ENTRY = 0.5                  # the entering pin passes this far left of where the rounded tip stops steering it left
M_DROP = 0.5                   # push 2: the pin leaving the cusp lands this far right on the ramp, past the rounded
                               # ridge top (so it can only go right, into V2)
M_V1, M_V2 = 0.45, 0.5         # pockets clear of the L / R corner arcs (each release steers the right way)
RIDGE_UP = 0.6                 # the ridge's rounded top stands this far above the V1 opening
THROAT_E = 0.3                 # throat half-width (pin centre, before the corner fillets widen it)


def _unit(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]; L = math.hypot(dx, dy); return dx / L, dy / L


def _offset_line(a, b, r, side):
    """line a-b moved by r to the left (+1) or right (-1) of its direction: (point, unit direction)"""
    tx, ty = _unit(a, b); nx, ny = (-ty, tx) if side > 0 else (ty, -tx)
    return (a[0] + r * nx, a[1] + r * ny), (tx, ty)


def _meet(l0, l1):
    (p0, d0), (p1, d1) = l0, l1
    den = d0[0] * d1[1] - d0[1] * d1[0]
    s = ((p1[0] - p0[0]) * d1[1] - (p1[1] - p0[1]) * d1[0]) / den
    return p0[0] + s * d0[0], p0[1] + s * d0[1]


def _area(pts):
    return 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1]))


def island_ccw():
    return ISLAND if _area(ISLAND) > 0 else ISLAND[::-1]


def _out(a, b):
    """outward side of island edge a->b (island listed counter-clockwise -> outward is to the right)"""
    return -1 if _area(ISLAND) > 0 else +1


def _core():
    """the island moved in by RHO_I (sharp corners) = its fillet centres. The island is the core grown by RHO_I (arcs at
    the convex corners, the notch corner C stays sharp), so island grown by RP = core grown by RI exactly"""
    n = len(ISLAND); s = -_out(None, None)
    lines = [_offset_line(ISLAND[i], ISLAND[(i + 1) % n], RHO_I, s) for i in range(n)]
    return [_meet(lines[i - 1], lines[i]) for i in range(n)]


CORE = _core()
Tc, Kc, Lc, Cc, Rc, Bc, B2c = CORE


def grown_island(r=RI, res=256):
    from shapely.geometry import Polygon
    return Polygon(CORE).buffer(r, resolution=res)


# ---------------- derived configuration-space features ----------------
def latch_point(r=None):
    """where the pin centre rests when latched: the underside L->C (moved out) meets the notch wall C->R (moved out),
    or, when that contact would fall beyond R, the arc around R's fillet centre"""
    r = RI if r is None else r
    s = _out(None, None)
    l_lc, l_cr = _offset_line(Lc, Cc, r, s), _offset_line(Cc, Rc, r, s)
    q = _meet(l_lc, l_cr)
    t = _unit(Cc, Rc); seg = math.hypot(Rc[0] - Cc[0], Rc[1] - Cc[1])
    foot = (q[0] - Cc[0]) * t[0] + (q[1] - Cc[1]) * t[1]
    if 0.0 <= foot <= seg:
        return q
    (p0, d0) = l_lc
    fx, fy = p0[0] - Rc[0], p0[1] - Rc[1]
    b = 2 * (fx * d0[0] + fy * d0[1]); c = fx * fx + fy * fy - r * r
    ts = [(-b - math.sqrt(b * b - 4 * c)) / 2, (-b + math.sqrt(b * b - 4 * c)) / 2]
    cands = [(p0[0] + k * d0[0], p0[1] + k * d0[1]) for k in ts]
    return min(cands, key=lambda p: math.hypot(p[0] - q[0], p[1] - q[1]))


def derived():
    r = RI; s = _out(None, None)
    cusp = latch_point(r)
    (pL, dLC) = _offset_line(Lc, Cc, r, s)
    under_u0 = pL[0]                                   # left end of the straight underside (start of the L arc)
    (pR, dRB) = _offset_line(Rc, Bc, r, s)
    rb_u0 = pR[0]                                      # left end of the straight R->B edge (end of the R arc)
    u_lo = Lc[0] - r - GAP                             # left outer wall
    u_ro = max(Bc[0], B2c[0]) + r + GAP                             # right outer wall
    tip_top = Tc[1] - r
    # throat. Its right corner is a material wedge between the funnel wall (FUNNEL_WALL off -d) and the top wall
    # (MAX_WALL off +d), filleted. A pin sliding down the funnel wall follows the fillet arc (radius RO in pin-centre
    # space) and drops off it at its leftmost point, `reach` right of the sharp corner thr_r. That drop line must be
    # M_ENTRY left of u_lim, where the tip arc stops steering left within MAX_WALL.
    kfun = 1.0 / math.tan(math.radians(FUNNEL_WALL))
    k = 1.0 / T50
    wedge = math.radians(180 - FUNNEL_WALL - MAX_WALL)                 # angle of the material wedge
    bis = 0.5 * (math.radians(90 - MAX_WALL) - math.radians(90 - FUNNEL_WALL))   # bisector tilt from +u
    reach = RO / math.sin(wedge / 2) * math.cos(bis) - RO
    u_lim = Tc[0] - r * math.sin(math.radians(90 - MAX_WALL))
    thr_r = u_lim - M_ENTRY - reach
    thr_l = thr_r - 2 * THROAT_E
    from shapely.geometry import LineString
    grown = grown_island(r)
    d_thr = 1e9
    for i in range(401):
        u = thr_r + (u_ro - thr_r) * i / 400
        cut = grown.intersection(LineString([(u, -50), (u, 50)]))
        if not cut.is_empty:
            d_thr = min(d_thr, cut.bounds[1] - GAP - (u - thr_r) * k)
    d_ra = d_thr + (u_ro - thr_r) * k                  # where the top wall meets the right outer wall
    d_lb = d_thr + (thr_l - u_lo) * k                  # left wall reaches u_lo
    u0, u1 = u_lo - 0.2, u_ro + 0.2
    d_mouth = d_thr - max((thr_l - u0), (u1 - thr_r)) * kfun
    # floor. Push 1: ramp Q1 -> pocket V1 (under the straight underside). Push 2: from the cusp straight down onto
    # ramp P -> pocket V2 (right of the R corner). The ridge between the pockets is a FLOOR_WALL wedge with a filleted
    # top: in pin-centre space an arc of radius RO whose centre is RO right of the sharp config ridge P. The pin
    # dropping from the cusp must meet the straight ramp, M_DROP right of where that arc ends.
    hw = WELL_W / 2
    fw = math.radians(FLOOR_WALL)
    kf = 1.0 / math.tan(fw)
    u_p = cusp[0] - RO * (1 + math.cos(fw)) - M_DROP
    u_v1 = u_p - hw
    u_v2 = rb_u0 + M_V2 + hw
    if u_v1 - hw < under_u0 + M_V1:
        raise ValueError(f"pocket V1 at {u_v1:.2f} is not under the straight underside (starts {under_u0:.2f}): "
                         f"lengthen UNDER_LEN")

    def island_bottom(u):
        cut = grown.intersection(LineString([(u, -50), (u, 50)]))
        return None if cut.is_empty else cut.bounds[3]

    def deepest(f, ua, ub, n=400):
        """how far floor f (relative) must go down to stay GAP under the grown island between ua and ub"""
        out = -1e9
        for i in range(n + 1):
            u = ua + (ub - ua) * i / n; b = island_bottom(u)
            if b is not None:
                out = max(out, b + GAP - f(u))
        return out
    # left part: ramp from the left outer wall down to the V1 opening
    d_q1 = deepest(lambda u: (u - u_lo) * kf, u_lo, u_v1 - hw)
    d_v1 = d_q1 + (u_v1 - hw - u_lo) * kf
    # right part (relative to the ridge point P): the ridge's true rounded top, the ramp to V2, a 45 deg rise to the
    # right outer wall
    t_half = math.tan(fw / 2)
    cu, cd = u_p + RO, RO / t_half                      # the ridge fillet centre in pin-centre space
    u_t = cu + RO * math.cos(fw)
    d_v2r = (u_v2 - hw - u_p) * kf
    def right_floor(u):
        if u <= u_t:
            return cd - math.sqrt(max(0.0, RO * RO - (u - cu) ** 2))
        if u <= u_v2 - hw:
            return (u - u_p) * kf
        if u <= u_v2 + hw:
            return d_v2r
        return d_v2r - (u - u_v2 - hw)
    d_p = max(cusp[1] + 0.9 - (cusp[0] - u_p) * kf,     # the pin leaving the cusp lands on the ramp >= 0.9 lower
              deepest(right_floor, u_p, u_ro))
    d_v2 = d_p + d_v2r
    d_q2 = d_v2 - (u_ro - u_v2 - hw)
    ridge_top = d_p + cd - RO
    d_v1 = max(d_v1, ridge_top + RIDGE_UP)              # the ridge stands RIDGE_UP above the V1 opening
    d_top = max(d_v1, d_v2)
    d_v, d_stop, d_well = d_top, d_top + STOP_IN_WELL, d_top + WELL_D
    return dict(cusp=cusp, under_u0=under_u0, rb_u0=rb_u0, u_lo=u_lo, u_ro=u_ro, tip_top=tip_top, d_thr=d_thr,
                d_ra=d_ra, d_lb=d_lb, u0=u0, u1=u1, d_mouth=d_mouth, u_v1=u_v1, u_p=u_p, u_v2=u_v2, d_q1=d_q1,
                d_v=d_v, d_p=d_p, d_q2=d_q2, thr_l=thr_l, thr_r=thr_r, d_stop=d_stop, d_well=d_well,
                d_v1=d_v1, d_v2=d_v2, u_lim=u_lim, reach=reach, ridge_top=ridge_top)


def outer():
    """outer limit of the pin centre (before the fillets), counter-clockwise in (u, d)"""
    D = derived()
    hw = WELL_W / 2
    u1_, u2_ = D["u_v1"], D["u_v2"]
    pts = [(D["u0"], D["d_mouth"]), (D["thr_l"], D["d_thr"]), (D["u_lo"], D["d_lb"]), (D["u_lo"], D["d_q1"]),
           (u1_ - hw, D["d_v1"]), (u1_ - hw, D["d_well"]), (u1_ + hw, D["d_well"]), (u1_ + hw, D["d_p"]),
           (u2_ - hw, D["d_v2"]), (u2_ - hw, D["d_well"]), (u2_ + hw, D["d_well"]),
           (u2_ + hw, D["d_v2"]), (D["u_ro"], D["d_q2"]),
           (D["u_ro"], D["d_ra"]), (D["thr_r"], D["d_thr"]), (D["u1"], D["d_mouth"])]
    return pts if _area(pts) > 0 else pts[::-1]


def island_outline():
    """the physical island: the core grown by RHO_I (lines + fillet arcs; the notch corner C stays sharp)"""
    core = CORE if _area(CORE) > 0 else CORE[::-1]
    return offset_ring(core, RHO_I)


def island_polygon():
    from shapely.geometry import Polygon
    return Polygon(elements_to_points(island_outline()))


def free_polygon(delta=0.0, res=256):
    """where the pin centre can really go: the physical cut shrunk by the pin radius (+ delta: a pin that much
    bigger, or a print whose walls came out that much fat). This, not outer(), is what the simulations run on: the
    shrink rounds every corner that sticks into the track (throat edges, ridge top)"""
    return track_polygon().buffer(-(RP + delta), resolution=res)


# ---------------- exact outer wall: OUTER grown by RP, inward corners filleted ----------------
def offset_ring(pts, r, fillet=0.0):
    """grow the region bounded by ring `pts` (region on the LEFT of the ring) by r; at concave corners of the region
    (the walls that stick in) put a fillet of radius `fillet`.
    Elements: ('line', p, q) and ('arc', centre, p, q, mid), in order around the ring."""
    n = len(pts); edges = []
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        p, t = _offset_line(a, b, r, -1)
        edges.append((p, (p[0] + (b[0] - a[0]), p[1] + (b[1] - a[1])), t, (t[1], -t[0])))
    starts, ends, joins = [None] * n, [None] * n, [None] * n
    for i in range(n):
        e0, e1 = edges[i - 1], edges[i]; t0, t1 = e0[2], e1[2]
        if t0[0] * t1[1] - t0[1] * t1[0] > 1e-12:     # left turn: convex corner of the region -> arc
            v = pts[i]; bis = (e0[3][0] + e1[3][0], e0[3][1] + e1[3][1]); L = math.hypot(*bis)
            mid = (v[0] + r * bis[0] / L, v[1] + r * bis[1] / L)
            ends[i - 1], starts[i] = e0[1], e1[0]; joins[i] = ("arc", v, e0[1], e1[0], mid)
        elif fillet > 0:                               # concave corner: fillet centred where the walls offset by
            n0, n1 = e0[3], e1[3]                      # r + fillet meet
            c = _meet(_offset_line(pts[i - 1], pts[i], r + fillet, -1), _offset_line(pts[i], pts[(i + 1) % n], r + fillet, -1))
            p = (c[0] - fillet * n0[0], c[1] - fillet * n0[1]); q = (c[0] - fillet * n1[0], c[1] - fillet * n1[1])
            bis = (n0[0] + n1[0], n0[1] + n1[1]); L = math.hypot(*bis)
            mid = (c[0] - fillet * bis[0] / L, c[1] - fillet * bis[1] / L)
            ends[i - 1], starts[i] = p, q; joins[i] = ("arc", c, p, q, mid)
        else:
            x = _meet((e0[0], t0), (e1[0], t1)); ends[i - 1] = starts[i] = x
    out = []
    for i in range(n):
        if joins[i]:
            out.append(joins[i])
        p, q, t = starts[i], ends[i], edges[i][2]
        if (q[0] - p[0]) * t[0] + (q[1] - p[1]) * t[1] <= 1e-9:
            raise ValueError(f"offset edge {i} {pts[i]} -> {pts[(i + 1) % n]} vanishes at r = {r}")
        out.append(("line", p, q))
    return out


def outer_outline(r=RP):
    return offset_ring(outer(), r, RHO_O)


def arc_points(c, p, q, mid, n=48):
    a0 = math.atan2(p[1] - c[1], p[0] - c[0]); a1 = math.atan2(q[1] - c[1], q[0] - c[0])
    am = math.atan2(mid[1] - c[1], mid[0] - c[0]); tau = 2 * math.pi
    sweep = (a1 - a0) % tau
    if not ((am - a0) % tau < sweep):
        sweep -= tau
    rr = math.hypot(p[0] - c[0], p[1] - c[1])
    return [(c[0] + rr * math.cos(a0 + sweep * k / n), c[1] + rr * math.sin(a0 + sweep * k / n)) for k in range(n)]


def elements_to_points(els):
    pts = []
    for e in els:
        pts += [e[1]] if e[0] == "line" else arc_points(*e[1:])
    return pts


def track_polygon():
    """the physical cut in the track plane"""
    from shapely.geometry import Polygon
    return Polygon(elements_to_points(outer_outline())).difference(island_polygon())


def margins(delta=0.0):
    """distances at each decision point, measured on the real pin-centre space (free_polygon), for a pin `delta`
    bigger than designed. All should be clearly positive."""
    from shapely.geometry import LineString, Point
    D = derived(); F = free_polygon(delta)
    r = RI + delta
    out = {}
    # entry: the pin comes down the right funnel wall and drops off the throat's right fillet; furthest-right drop
    # line = rightmost free u in a thin band just below the throat corner, left of the tip
    band = F.intersection(LineString([(D["thr_l"] - 3, D["d_thr"] + 0.05), (Tc[0], D["d_thr"] + 0.05)]))
    u_drop = max(x for g in getattr(band, "geoms", [band]) for x, _ in g.coords)
    u_lim = Tc[0] - r * math.sin(math.radians(90 - MAX_WALL))
    out["entry (drop line left of tip limit)"] = u_lim - u_drop
    # lip: how far the latched pin must sink to slide off the notch past R (the arc bottom under R's fillet centre)
    c = latch_point(r)
    out["notch lip (sink to escape)"] = Rc[1] + r - c[1]
    # push 2: the pin dropping from the cusp meets the ridge/ramp; it must be on the straight ramp, past the ridge arc
    out["push 2 drop past ridge top"] = c[0] - (D["u_p"] + RO + (RO + delta) * math.cos(math.radians(FLOOR_WALL)))
    out["V1 right of underside start"] = D["u_v1"] - WELL_W / 2 - _offset_line(Lc, Cc, r, _out(0, 0))[0][0]
    out["V2 right of R arc end"] = D["u_v2"] - WELL_W / 2 - _offset_line(Rc, Bc, r, _out(0, 0))[0][0]
    return out


if __name__ == "__main__":
    D = derived()
    for k, v in D.items():
        print(f"  {k:9s} {v if not isinstance(v, (float, int)) else round(v, 4)}")
    from shapely.geometry import Polygon
    T = track_polygon()
    B = Polygon(outer()).buffer(RO, resolution=256).buffer(-RHO_O, resolution=256).difference(
        Polygon(CORE).buffer(RHO_I, resolution=256))
    print("track valid", T.is_valid, "area", round(T.area, 4), "vs shapely", round(B.area, 4),
          "sym diff", round(T.symmetric_difference(B).area, 6))
    print("track bounds (u0, d0, u1, d1):", [round(x, 3) for x in T.bounds], " width", round(T.bounds[2] - T.bounds[0], 2),
          "length", round(T.bounds[3] - T.bounds[1], 2))
    I = island_polygon(); print("island area", round(I.area, 3), " eroded by 0.6 (1.2 mm wall) still one piece:",
                                I.buffer(-0.6).geom_type, round(I.buffer(-0.6).area, 3))
    for dl in (-0.1, 0.0, 0.1):
        print(f"margins, pin {dl:+.1f}:", {k: round(v, 3) for k, v in margins(dl).items()})
