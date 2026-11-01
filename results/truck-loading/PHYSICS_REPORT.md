# Truck end-to-end PhysX ranking

Every run derived commands only from RGB-D observations. Open-loop controllers froze the complete plan before motion; `vision_replan` refreshed bin/truck poses before transport; `vision_recovery` additionally retried camera-observed cubes still outside their matching bin. Object transforms were unavailable to every controller and read only after the final retreat.

| Rank | Policy | Controller | Full success | Failures | Cubes | Bins | Perception | Actions | TCP path | Motion time | Efficiency | Balanced |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `locked_calibration__depth_table_h6mm_s1_rawcube` | `vision_recovery_v3` | 6/6 (100.0%) | 0 | 12.00/12 | 3.00/3 | 1066.2 ms | 174.0 | 12.81 m | 28.9 s | 0.4213 | 0.7921 |

A passing Isaac run reduces risk but cannot guarantee physical success. Real camera/tool calibration, finger geometry, compliance, friction, controller timing, and unmodelled obstacles remain transfer variables.
