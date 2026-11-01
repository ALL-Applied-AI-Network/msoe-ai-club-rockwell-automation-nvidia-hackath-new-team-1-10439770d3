# Evidence-grounded assembly instruction

**Task:** Transfer four red/brown cubes into one green open-top bin

Six separate takes show the same result: four cubes are moved one at a time into the green bin. The video proves task order and qualitative contact states; it does not provide robot coordinates.

## Robot sequence

1. Detect and assign persistent IDs to four cubes and the target bin.
2. Select one unplaced cube; approach vertically with the complete finger keep-out checked.
3. Close once, hold closed, and perform a short proof lift.
4. Travel at safe height to a pose above the current bin opening.
5. Re-detect the bin, then descend on its opening axis under collision guarding.
6. Open once and hold open; retract vertically before moving laterally.
7. Confirm that the inside-bin count increased and repeat until it reaches four.

## Non-negotiable calibration

- calibrate RGB-D to robot base before resolving any pixel coordinate into motion
- model the complete installed fingers, not only TCP or jaw-center clearance
- proof-lift after closing; abort if tracked cube does not follow
- re-detect bin immediately before loaded descent
- use one debounced open/close command; reject oscillating command streams
- post-release occupancy increment is mandatory before advancing

## Evidence summary

- 6 independent episodes
- 24 successful demonstrated placements
- Grounded action boundaries reviewed to approximately ±0.25–0.5 s
- Metric 6D motion intentionally withheld until calibrated RGB-D/CAD is supplied
