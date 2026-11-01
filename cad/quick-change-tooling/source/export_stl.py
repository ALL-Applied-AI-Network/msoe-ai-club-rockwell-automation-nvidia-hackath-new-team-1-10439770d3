"""export_stl.py: fine STLs for printing, from the STEP exports (out/*.STEP -> out/*.STL).

SolidWorks' default STL is coarse: a 1.4 mm steering arc in the track becomes a few flat facets the peg catches on.
Here every surface is within 0.005 mm of the true one and every arc is cut into facets of at most 3 deg.

    python export_stl.py [names...]        (default: every printed latch part)
"""
import os, sys
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
import check_latch as C

PRINTED = ["Latch_shuttle", "Receiver_Z", "Finger_Z_flat", "Finger_Z_V", "Receiver_Y", "Finger_Y_flat", "Finger_Y_V",
           "Latch_test_base", "Latch_test_key"]
from finger_tips import TIPS
PRINTED += [f"Finger_{v}_{t}" for v in ("Z", "Y") for t in TIPS] + ["Rack_Z_N6", "Rack_Z_W3"]


def export(name, lin=0.005, ang=0.05):
    s = C.load(name)
    BRepMesh_IncrementalMesh(s, lin, False, ang, True)
    w = StlAPI_Writer(); w.ASCIIMode = False
    path = os.path.join(C.OUT, name + ".STL")
    ok = w.Write(s, path)
    print(f"  {name}.STL  {os.path.getsize(path) / 1e6:.1f} MB  {'ok' if ok else 'FAILED'}")


if __name__ == "__main__":
    for n in sys.argv[1:] or PRINTED:
        if os.path.exists(os.path.join(C.OUT, n + ".STEP")):
            export(n)
