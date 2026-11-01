"""simulate.py: run the pin through the push-push cycle in the track and check every outcome.

Model: the pin centre moves in FREE. The finger drives it along d (+1 push, -1 release by the eject spring); it is free
in u (shuttle). On contact with a wall it slides along the wall if the wall is shallow enough for the friction
(|t.m| > mu * |n.m|, mu = track + shuttle-slot friction together); otherwise it stops. A push ends when the pin is
stopped (a valley); a release ends when the pin is stopped (the cusp) or leaves through the mouth.

Runs on the REAL pin-centre space: the physical cut (with its fillets) shrunk by the pin radius.
Checked for: every start position across the funnel mouth, friction 0 / 0.4 / 0.7 / 0.8, the pin (or the track)
0.1 mm larger and smaller than designed, the hard stop off by print error, and the shuttle knocked 0.2 mm sideways
before every stroke.
"""
import math, sys
from shapely.geometry import Polygon, Point, LineString
from shapely.prepared import prep
import track_geom as TG

H = 0.004
JOLTS = [0, 0, 0]                                    # knocks applied in full / in part (wall in the way) / not at all


def free_for(delta=0.0):
    """where the pin centre can really go when it is `delta` larger in radius than designed (or the track walls that
    much fat): the physical cut shrunk by the pin radius"""
    return TG.free_polygon(delta, res=32).simplify(0.0005)


class Sim:
    def __init__(self, F):
        if F.geom_type == "MultiPolygon":                  # drop slivers left by the mitred erosion
            F = max(F.geoms, key=lambda g: g.area)
        self.F = F; self.P = prep(F)
        segs = []
        for ring in [F.exterior] + list(F.interiors):
            c = list(ring.coords)
            segs += [(c[i], c[i + 1]) for i in range(len(c) - 1)]
        self.segs = segs
        from shapely.strtree import STRtree
        self.tree = STRtree([LineString(sg) for sg in segs])

    def inside(self, q):
        return self.P.covers(Point(q))

    def near_segments(self, q, k=4):
        """unit tangents of the walls within reach of q, nearest first"""
        idx = self.tree.query(Point(q).buffer(4 * H))
        out = []
        for i in idx:
            (ax, ay), (bx, by) = self.segs[i]; dx, dy = bx - ax, by - ay; L = math.hypot(dx, dy)
            dist = LineString(self.segs[i]).distance(Point(q))
            out.append((dist, (dx / L, dy / L)))
        out.sort()
        return [t for _, t in out[:k]]

    def run(self, p, m, mu, d_exit, max_steps=20000, d_stop=None, jolt=0.0, jolt_d=None):
        """d_stop: the finger's hard stop (a push ends there, as it does on the real part).
        jolt: knock the shuttle this far sideways when the pin passes depth jolt_d (as far as the walls allow)"""
        path = [p]; last_t = None
        pending = bool(jolt) and jolt_d is not None
        for _ in range(max_steps):
            if pending and (p[1] - jolt_d) * m[1] >= 0:
                pending = False
                for f in (1.0, 0.75, 0.5, 0.25):
                    q = (p[0] + f * jolt, p[1])
                    if self.inside(q):
                        p = q; path.append(p); JOLTS[0 if f == 1.0 else 1] += 1; break
                else:
                    JOLTS[2] += 1
            if m[1] < 0 and p[1] < d_exit:
                return p, "out", path
            if m[1] > 0 and d_stop is not None and p[1] >= d_stop:
                p = (p[0], d_stop); path[-1] = p                 # the stop face, not a step past it
                return p, "stop", path
            q = (p[0] + H * m[0], p[1] + H * m[1])
            if self.inside(q):
                p = q; path.append(p); last_t = None; continue
            cands = self.near_segments(q) + ([last_t] if last_t else [])
            moved = False; jam = True
            for t in cands:
                tm = t[0] * m[0] + t[1] * m[1]
                if tm < 0:
                    t = (-t[0], -t[1]); tm = -tm
                nm = math.sqrt(max(0.0, 1 - tm * tm))
                if tm <= mu * nm + 1e-9:
                    continue                                 # this wall holds the pin by friction
                jam = False
                for s_ in (1.0, 0.5, 0.25):
                    q2 = (p[0] + s_ * H * t[0], p[1] + s_ * H * t[1])
                    if self.inside(q2):
                        p = q2; path.append(p); moved = True; last_t = t; break
                if moved:
                    break
            if not moved:
                return p, "stuck", path
        return p, "limit", path


def classify(p, D, delta=0.0):
    """where a stroke ended: in pocket V1 / V2 (the straight slots under the valleys), at the latch point, or neither"""
    hw = TG.WELL_W / 2 + abs(delta) + 0.05
    if abs(p[0] - D["u_v1"]) <= hw and p[1] >= D["d_v1"] - 0.05:
        return "V1"
    if abs(p[0] - D["u_v2"]) <= hw and p[1] >= D["d_v2"] - 0.05:
        return "V2"
    c = TG.latch_point(TG.RI + delta)
    if math.hypot(p[0] - c[0], p[1] - c[1]) < 0.3:
        return "cusp"
    return f"?({p[0]:.2f},{p[1]:.2f})"


def cycle(sim, u_start, mu, D, delta=0.0, d_stop=None, jolt=(0.0, 0.0, 0.0, 0.0)):
    """insert/push 1 -> pocket V1, release -> latch point, push 2 -> pocket V2, release -> out of the mouth.
    Pushes end at the hard stop d_stop (default: as designed, in the middle of the pockets' common window)."""
    d_stop = D["d_stop"] if d_stop is None else d_stop
    d0 = D["d_mouth"] + max(delta, 0.0)              # the mouth edge of this configuration space
    d_exit = d0 + 0.03                               # back at the mouth edge = the finger is free
    p = (u_start, d0 + 0.02)
    res, paths = [], []
    # where each stroke makes its decision; the shuttle is knocked just before it (play, a fast hand)
    jolt_at = (D["d_thr"] + 0.3, D["d_v1"] - 0.3, D["cusp"][1] + 0.3, D["d_v2"] - 0.3)
    for k, (m, want) in enumerate((((0, 1), "V1"), ((0, -1), "cusp"), ((0, 1), "V2"), ((0, -1), "out"))):
        p, how, path = sim.run(p, m, mu, d_exit, d_stop=d_stop, jolt=jolt[k], jolt_d=jolt_at[k])
        got = "out" if how == "out" else (classify(p, D, delta) if how in ("stuck", "stop") else how)
        res.append(got); paths.append(path)
        if k % 2 and got != want:                    # a push may end anywhere; what counts is where each release ends
            return False, res, paths
    return True, res, paths


def stroke_tolerance(deltas=(0.1, 0.0, -0.15, -0.25), mus=(0.4, 0.8), n_starts=9, verbose=True):
    """the same cycle with the hard stop anywhere from the top to near the bottom of the pockets: how far the real
    stroke may be off (print error) and still latch and release"""
    D = TG.derived(); ok_all = True
    offs = [-TG.STOP_IN_WELL + x for x in (0.02, 0.3, 0.6, 0.9, 1.2, 1.5, TG.WELL_D - 0.1)]
    for delta in deltas:
        sim = Sim(free_for(delta))
        for mu in mus:
            bad = []
            for off in offs:
                for i in range(n_starts):
                    u = D["u0"] + 0.05 + (D["u1"] - D["u0"] - 0.1) * i / (n_starts - 1)
                    if not sim.inside((u, D["d_mouth"] + max(delta, 0.0) + 0.02)):
                        continue
                    ok, res, _ = cycle(sim, u, mu, D, delta, D["d_stop"] + off)
                    if not ok:
                        bad.append((round(off, 2), round(u, 2), res))
            ok_all &= not bad
            if verbose:
                print(f"  stroke error {offs[0]:+.2f} .. {offs[-1]:+.2f} mm, pin {delta:+.2f}, mu {mu}: "
                      + ("all OK" if not bad else f"{len(bad)} FAIL e.g. {bad[0]}"), flush=True)
    return ok_all


def jolt_test(deltas=(-0.1, 0.0, 0.1), mus=(0.0, 0.4, 0.8), j=0.2, n_starts=9, verbose=True):
    """the shuttle knocked j mm left or right just before each stroke's decision point (all 16 combinations): the
    outcome must not change. Catches decisions that sit on a knife edge (which a slow, perfect stroke would hide)"""
    import itertools
    D = TG.derived(); ok_all = True
    for delta in deltas:
        sim = Sim(free_for(delta))
        for mu in mus:
            bad = []; n = 0
            for i in range(n_starts):
                u = D["u0"] + 0.05 + (D["u1"] - D["u0"] - 0.1) * i / (n_starts - 1)
                if not sim.inside((u, D["d_mouth"] + max(delta, 0.0) + 0.02)):
                    continue
                for js in itertools.product((-j, j), repeat=4):
                    n += 1
                    ok, res, _ = cycle(sim, u, mu, D, delta, jolt=js)
                    if not ok:
                        bad.append((round(u, 2), js, res))
            ok_all &= not bad
            if verbose:
                print(f"  jolt +-{j} mm, pin {delta:+.1f}, mu {mu}: {n - len(bad)}/{n} OK  (knocks so far: full / "
                      f"partial / blocked = {JOLTS})"
                      + (f"   FAIL e.g. {bad[0]}" if bad else ""), flush=True)
    return ok_all


def check_all(verbose=True):
    D = TG.derived()
    ok_all = True; worst = []
    for delta in (0.0, -0.1, 0.1):
        sim = Sim(free_for(delta))
        for mu in (0.0, 0.4, 0.7, 0.8):
            fails = []; n = 0
            u0, u1 = D["u0"] + 0.05, D["u1"] - 0.05
            for i in range(41):
                u = u0 + (u1 - u0) * i / 40
                if not sim.inside((u, D["d_mouth"] + max(delta, 0.0) + 0.02)):
                    continue
                n += 1
                ok, res, _ = cycle(sim, u, mu, D, delta)
                if not ok:
                    fails.append((round(u, 2), res))
            ok_all &= not fails
            if verbose:
                print(f"  pin {'+' if delta >= 0 else ''}{delta:.1f} mm, mu {mu:.1f}: {n - len(fails)}/{n} starts OK"
                      + (f"   FAIL e.g. {fails[0]}" if fails else ""), flush=True)
    return ok_all


def strokes():
    D = TG.derived()
    return dict(latch=D["cusp"][1] - D["d_mouth"], overtravel=D["d_stop"] - D["cusp"][1], total=D["d_stop"] - D["d_mouth"])


if __name__ == "__main__":
    D = TG.derived()
    s = strokes()
    print(f"strokes (mm): mouth->latched {s['latch']:.2f}, latched->hard stop (push-past) {s['overtravel']:.2f}, "
          f"total {s['total']:.2f}")
    ok = check_all()
    print("ALL CYCLES OK" if ok else "FAILURES")
    print("stroke tolerance (hard stop off by print error):")
    ok2 = stroke_tolerance()
    print("STROKE TOLERANCE OK" if ok2 else "STROKE TOLERANCE FAILURES")
    print("shuttle knocked sideways before each stroke:")
    ok3 = jolt_test()
    print("JOLT TEST OK" if ok3 else "JOLT TEST FAILURES")
    if "--plot" in sys.argv:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        sim = Sim(free_for(0.0))
        fig, ax = plt.subplots(figsize=(8, 11))
        T = TG.track_polygon()
        for ring in [T.exterior] + list(T.interiors):
            x, y = ring.xy; ax.fill(x, y, color="#dde6f0" if ring is T.exterior else "white", zorder=0)
            ax.plot(x, y, "k-", lw=1.2)
        F = sim.F
        for ring in [F.exterior] + list(F.interiors):
            x, y = ring.xy; ax.plot(x, y, "--", color="gray", lw=0.7)
        cols = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
        names = ["push 1", "release 1 (latch)", "push 2", "release 2 (out)"]
        _, _, paths = cycle(sim, D["u1"] - 0.3, 0.7, D)
        for c, nme, path in zip(cols, names, paths):
            xs, ys = zip(*path); ax.plot(xs, ys, color=c, lw=2, label=nme)
        _, _, paths = cycle(sim, D["u0"] + 0.3, 0.7, D)
        for c, path in zip(cols, paths):
            xs, ys = zip(*path); ax.plot(xs, ys, color=c, lw=1, ls=":")
        circ = plt.Circle(D["cusp"], TG.PIN_D / 2, color="red", alpha=0.4); ax.add_patch(circ)
        ax.invert_yaxis(); ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="lower right")
        ax.set_xlabel("u (shuttle direction), mm"); ax.set_ylabel("d (insertion depth), mm")
        ax.set_title("Heart track: physical cut (solid), pin-centre limits (dashed), pin at latch (red)")
        plt.tight_layout(); plt.savefig("track_cycle.png", dpi=110)
        print("saved track_cycle.png")
