"""simulate.py's check at the REAL pin size: the track is sized for a 2.3 mm pin (2.0 + 2 x 0.15 clearance), so the
real 2.0 mm pin has 0.15 mm extra play: delta = -0.15, plus +-0.1 mm print error."""
import simulate as S, track_geom as TG
D = TG.derived(); bad = 0
for delta in (-0.25, -0.15, -0.05):
    sim = S.Sim(S.free_for(delta))
    for mu in (0.0, 0.4, 0.8):
        n = f = 0; ex = None
        for i in range(41):
            u = D["u0"] + 0.05 + (D["u1"] - D["u0"] - 0.1) * i / 40
            if not sim.inside((u, D["d_mouth"] + 0.02)):
                continue
            n += 1; ok, res, _ = S.cycle(sim, u, mu, D, delta)
            if not ok:
                f += 1; ex = (round(u, 2), res)
        bad += f
        print(f"effective pin delta {delta:+.2f} mm, mu {mu}: {n - f}/{n} OK" + (f"  e.g. {ex}" if ex else ""), flush=True)
print("REAL-PIN CHECK:", "ALL OK" if not bad else f"{bad} failures")
