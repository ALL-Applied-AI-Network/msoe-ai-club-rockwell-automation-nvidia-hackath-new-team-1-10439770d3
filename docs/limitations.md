# Limitations and next validation

## Current limitations

- **Calibration dependent.** Camera extrinsics, table plane, TCP, tool frame, finger geometry, and printed part scale must agree.
- **Depth resolution.** The tested RGB-D streams were useful for table-relative height and coarse re-centering, but assemblies near 0.5 mm tolerance need better depth, fiducials, or local metrology.
- **Human gated.** The physical trials used approved sessions and operator supervision. The project does not claim unattended production autonomy.
- **Limited physical sample size.** The archive is rich in telemetry and video but does not yet contain a matched, repeated factory reliability study across many variants.
- **Simulation gap.** PhysX cannot fully reproduce pneumatic timing, printed-surface friction, compliance, cable effects, exposure changes, structural flex, or every contact failure.
- **Reconstruction scope.** The physical orbit observed an upper hemisphere; hidden faces were not measured. Outputs are engineering references, not metrology-certified CAD.
- **Quick-change prototype.** The latch concept needs retention-force, fatigue, contamination, sensing, and safety qualification.

## Recommended next experiment

Run a controlled 5–10-variant pilot with repeated trials and a fixed baseline. Measure:

1. engineering hours from new-part receipt to approved plan;
2. last-good-to-first-good cell downtime;
3. first-pass success and intervention count;
4. scrap and rework;
5. grasp/placement error by part and lighting condition;
6. calibration time and drift;
7. proof-lift false accepts/rejects; and
8. tool-change time and retention reliability.

Use improved close-range depth or fiducials for precision insertions, automatically log every plan version and approval, and reserve a blind final test set of parts/poses.

## Illustrative value model

The final submission included a ten-cell illustrative capacity model: 180 annual changeovers, 1,800 engineering hours, 225 cell-hours, and $490,500 gross annual capacity at the stated assumptions. It is a scenario for sizing a pilot, not a measured plant saving. A real study must replace all rates and changeover counts with plant data.

[Back to documentation](README.md)

