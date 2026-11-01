# Reverse-engineering experiment contract

## Objective

Recover usable `.obj` and `.stl` geometry for `gearboxopen.stl` and
`gearboxclosed.stl` from a robot-style orbit in Isaac Sim, without allowing a
reconstruction method to read the source mesh. The supplied meshes are used
only by capture and evaluation.

## Observation contract

Each method may consume:

- masked RGB frames at 960 x 720;
- noisy, quantized RGB-D observations;
- reported camera poses containing bounded FK/hand-eye error;
- camera intrinsics and timestamps;
- four simulated lighting observations for the photometric-stereo test;
- the declared relative pose of a second, flipped scan pass; and
- simulated touch coordinates only for the touch-probe experiment.

The evaluator-only directory contains the clean depth, true poses, and source
triangles. Reconstruction programs do not import that directory.

## Capture

- 18 azimuths x 5 elevations = 90 views per pass.
- Upright plus flipped pass = 180 observations per part.
- Nominal radius: 420 mm for the sensor-ceiling experiment.
- A separate ER-4iA inverse-kinematics gate tests practical radii and camera
  offsets. It does not claim collision or cable certification.

## Tested methods and fidelity labels

1. `neural_sdf`: compact Fourier-feature neural SDF trained from the noisy
   RGB-D surface samples. It is an empirical neural-implicit baseline, not a
   false claim of running Neuralangelo, NeuS2, or BakedSDF unchanged.
2. `laser_stripe_poisson`: idealized five-pixel line-depth ribbons followed by
   Poisson reconstruction. It estimates the value of line profiling but is not
   equivalent to a physical metrology laser.
3. `tsdf_fk` and `tsdf_icp`: fixed-FK and bounded pose-refined RGB-D TSDF.
4. Official 2D Gaussian Splatting: attempted separately on the prepared
   known-pose COLMAP dataset because it has its own CUDA build environment.
5. `dual_pass_tsdf`: upright plus flipped/in-hand-style observations with
   known relative pose.
6. `nbv_tsdf`: a view-novelty/coverage proxy selecting 24 observations. It is
   not described as a full online OctoMap entropy controller.
7. `photometric_poisson`: four-light normal estimation fused with depth. The
   lights are simulated; the present physical cell has not demonstrated this
   lighting rig.
8. `visual_hull`: RGB masks, camera geometry, voxel space carving, and no
   per-pixel depth reconstruction.
9. `touch_scaled_tsdf`: simulated six-axis contact bounds used to correct the
   optical mesh extents. It is a contact-constrained proxy, not full ARAP mesh
   deformation.
10. Official COLMAP MVS: attempted separately with the reported known camera
    poses and masked RGB images.

## Metrics

The source models are multi-component and non-watertight. Consequently, the
primary metrics are bidirectional sampled surface distance, F-score at
0.5/1/2/5 mm, normal consistency, dimensional error, bottom-surface recall,
and runtime. Watertightness is reported as a one-point tie-breaker rather than
treated as ground truth.

