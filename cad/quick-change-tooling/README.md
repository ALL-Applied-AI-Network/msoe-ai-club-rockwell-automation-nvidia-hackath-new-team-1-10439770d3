# Push-push quick-change tooling

The concept turns a replaceable finger into a push-to-latch, push-again-to-release module.

## Included

- receiver, shuttle, test coupon, key, and two rack concepts;
- a Y-orientation engineering tip catalog in STEP and STL;
- parametric/generation and geometry-check scripts; and
- representative preview images under [`media/images`](../../media/images/).

The most presentation-focused engineering examples are the flat screwdriver, 3 mm hex, PH2, and 7 mm socket tips.

## Source requirements

The geometry check and rendering scripts use OpenCascade (`OCP`), NumPy, trimesh, and matplotlib. The native SolidWorks generator additionally expects the team's small `swkit2` automation helper on `ATEAM_CAD_HELPERS` and a licensed local SolidWorks installation. The exported STEP/STL models are included so the design can still be inspected and printed without that proprietary CAD environment.

## Prototype boundary

The geometry is not production qualified. Before use on a moving industrial robot, test retention force, fatigue, accidental release, contamination sensitivity, spring life, jaw attachment, sensing, payload, and fail-safe behavior.

