"""check_asm.py: read back out/Gripper_latch_<v>.STEP (SolidWorks' own assembly export) and compare every solid with
the parts placed the way check_latch.py places them (latched, full open): bounding boxes and volumes."""
import os, sys
import check_latch as C
import latch_layout as L
from analyze_step import solids

def main(var):
    v = L.VARIANTS[var]
    got = solids(os.path.join(C.OUT, f"Gripper_latch_{var}.STEP"))
    body, jaw = C.load("MHZL2-16D2_body"), C.load("MHZL2-16D2_jaw")
    rec, fin, sh = C.load(f"Receiver_{var}"), C.load(f"Finger_{var}_flat"), C.load("Latch_shuttle")
    S0 = C.shuttle_at(sh, v, L.U_C)
    want = [("body", body), ("jaw A", jaw), ("jaw B", C.rotz180(jaw)), ("receiver A", rec), ("receiver B", C.rotz180(rec)),
            ("shuttle A", S0), ("shuttle B", C.rotz180(S0)), ("finger A", fin), ("finger B", C.rotz180(fin))]
    used, worst = set(), 0.0
    print(f"{var}: assembly STEP has {len(got)} solids, expected {len(want)}")
    for name, w in want:
        bw, vw = C.box(w), C.V(w)
        k = min((i for i in range(len(got)) if i not in used), key=lambda i: max(abs(a - b) for a, b in zip(C.box(got[i]), bw)) + abs(C.V(got[i]) - vw))
        used.add(k)
        d = max(abs(a - b) for a, b in zip(C.box(got[k]), bw)); worst = max(worst, d)
        print(f"   {name:10s} box difference {d:.5f} mm, volume {C.V(got[k]):10.3f} vs {vw:10.3f}")
    print(f"   worst placement difference {worst:.5f} mm")

for var in sys.argv[1:] or ["Y", "Z"]:
    main(var)
