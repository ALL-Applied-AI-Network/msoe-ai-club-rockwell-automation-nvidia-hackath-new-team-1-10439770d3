# Three Experiments: Dataset & Validation Strategy

> **Knowledge-base entry · Kickoff synthesis · The A-Team**  
> This document converts the three challenge descriptions into testable engineering plans. It is a decision aid, not a claim that a final dataset has already been selected.

## Executive decision

The three options reward different strengths:

| Experiment | Best reason to choose it | Primary risk | Current recommendation |
|---|---|---|---|
| **1. Final Vial Inspection** | Fastest path to an objective, factory-relevant baseline | Synthetic-to-real visual gap | **Start here unless the asset audit changes the evidence** |
| **2. Fluid Mixing Station 3** | Best causal and multimodal reasoning story | Only 30 labeled vials; color is optically difficult | Keep as the strongest alternative |
| **3. Grasping and Sorting** | Highest-impact live robotics demonstration | Calibration, safety, physical access, pending assets | Select only after a successful twin smoke test and confirmed test slot |

The A-Team should make the final selection through a one-session **asset and baseline gate**:

1. verify every download and permission;
2. enumerate files, labels, views, and timestamps;
3. visualize representative examples;
4. run the smallest baseline or simulator smoke test;
5. confirm a supervised physical-validation slot; and
6. choose the experiment whose most important risk can be tested first.

## Shared principles across all experiments

### Real data is evaluation evidence

The event guidance asks teams to inspect real-line data to understand the domain while creating training data with the supplied digital twins. Unless organizers explicitly approve otherwise, real samples should remain outside model training and be reserved for domain study, calibration decisions, and held-out evaluation.

### Avoid the easiest form of leakage

- Do not randomly split crops taken from the same full image.
- Do not split adjacent frames from one fill event across training and testing.
- Do not tune on the same physical robot positions later reported as unseen tests.
- Freeze the final test set before repeated augmentation or prompt iteration.

### Cost matters more than one accuracy number

A missed defect, unnecessary reject, incorrect cause, failed grasp, and safety intervention have different operational costs. Each experiment needs a cost-aware scorecard, not one aggregate accuracy value.

### Every final claim needs traceability

Record the dataset version, digital-twin version, generator settings, label source, split manifest, model or prompt version, decision threshold, runtime hardware, and random seed. Preserve failed runs and error examples.

---

## Experiment 1 — Final Vial Inspection

### Manufacturing problem

The final station contains two fixtures: **G** for good vials and **S** for scrap. Each fixture has 30 fixed numbered apertures. The robot fills the rightmost column from top to bottom, then moves left.

The overhead camera must determine the state of a target aperture:

| Label | Required interpretation |
|---|---|
| `pass` | Visible cap and vial body appear acceptable; hidden properties are not guaranteed |
| `missing_cap` | A vial is present, but its mouth is visibly open |
| `crooked_cap` | The cap is visibly tilted or not seated; a leaning vial alone is insufficient |
| `damaged_vial` | The target vial body is visibly damaged |
| `no_vial` | The target aperture is empty; whether this is an error depends on fill progress |

The unit of prediction is the **numbered target aperture**, not the entire crop. Neighboring transparent vials may overlap the crop and should not change the target label.

### Data contract

The description promises:

- full overhead images of both fixtures;
- labeled and unlabeled crops for numbered positions;
- fixture coordinates and position identifiers;
- source-image, fixture, and aperture mapping;
- a browser labeling/review tool with JSON export; and
- a digital twin for labeled synthetic images.

**Access:** [Final Vial Inspection dataset](https://drive.google.com/drive/folders/1Vl5qMLvUZDeWYNaFvnN8JbIqesjOKkWN?usp=sharing) · [Datasets 1 & 2 digital twin](https://drive.google.com/file/d/1nXRN_VZ_DKGS33jtELwn6z8WR8dbQfKe/view?usp=sharing)

### Hypotheses

**H1 — Fixed geometry is valuable.** Position-specific crops or reference images will outperform a single unconstrained full-scene classifier at the same data volume.

**H2 — Context reduces false alarms.** Combining crop-level appearance with fixture fill state will distinguish true missed placements from positions that are not yet expected to be occupied.

**H3 — Targeted randomization closes the gap.** Synthetic variation in camera pose, glare, cap seating, transparency, and lighting will improve real-scene recall more than simply generating additional ideal images.

### Baseline ladder

1. **Geometry baseline:** reconstruct and visualize all 60 target apertures for every scene.
2. **Golden-reference baseline:** compare each target crop with a confirmed-good reference and threshold a visual-difference score.
3. **Five-class baseline:** train a compact classifier on synthetic crops.
4. **Fixture context:** incorporate placement count/order-complete state and detect internal gaps.
5. **Full-scene consistency:** use neighboring positions to resolve overlapping transparent bodies and damaged-vial ambiguity.
6. **Failure-directed augmentation:** generate synthetic examples specifically matching the errors observed on held-out real data.

### Split and validation protocol

- Split by full source image, order, or collection session—not by crop.
- Keep real evaluation scenes untouched by training and prompt examples.
- Stratify the synthetic validation set by class, fixture, position, lighting, and pose.
- Test complete fixture scenes in addition to isolated crops.
- Inspect every false negative and a representative sample of false positives.

### Scorecard

| Metric | Why it matters |
|---|---|
| Recall for each defect class | Reveals which failures escape |
| Escaped defects / false negatives | Highest-priority judging and operational failure |
| False alarms / good-vial rejects | Measures unnecessary reinspection and rework |
| False-alarm rate at 100% known-defect recall | Gives an actionable operating point |
| Fixture occupancy accuracy | Tests missed-placement reasoning |
| Per-scene latency | Indicates whether full-fixture inspection is practical |
| Synthetic-to-real delta | Quantifies the domain gap honestly |

### Failure modes to attack early

- Transparent uncapped vial mistaken for `no_vial`.
- Glare or lighting change mistaken for a cap defect.
- Leaning vial mistaken for `crooked_cap`.
- Neighboring vial drives the crop prediction.
- Trailing empty position flagged before the order is complete.
- Synthetic defect looks too clean or physically implausible.

### Winning version

A two-level inspector: a visual model evaluates each numbered position while a fixture-state layer reasons about sequence and occupancy. The demonstration should show a full fixture, identify the exact position and defect, explain the confidence/threshold, and route uncertain cases to operator review.

---

## Experiment 2 — Fluid Mixing Station 3

### Manufacturing problem

Four tanks are present, but T-300 was out of service during collection. The usable configuration was:

| Tank | Dataset state |
|---|---|
| T-100 | Clear diluent |
| T-200 | Orange fluid |
| T-300 | Out of service; do not use |
| T-400 | Blue fluid |

Fluids share an upstream mixing path and single nozzle. When the recipe changes, residual liquid can carry into the next batch. The first vial can therefore be discolored while later vials flush clean. The provided example moves from an orange Recipe 6 to a blue Recipe 7: leftover orange and new blue plausibly create a green first vial.

The valuable output is not merely “different color.” It is a structured conclusion connecting the observed vial to the current recipe, previous recipe, tank state, dispense order, and probable carryover.

### Data contract

The description promises five synchronized video groups with three fixed 640×480 cameras at 30 fps and still images approximately every two seconds:

- `vial_fill`
- `tank_level`
- `cap_seal`

Thirty vials across ten recipes are labeled on `vial_fill`. Metadata includes recipe ID, per-tank volume, total volume, dispense order, fill start/end time, and per-frame filling state. Other views and runs are unlabeled. The twin covers fill and cap/seal views; tanks are not modeled and liquid color is approximate.

**Access:** [Fluid Mixing Station dataset](https://drive.google.com/drive/folders/1qwV2BEz3ODX17kwNCr9M15DfkY8U_blQ?usp=sharing) · [Datasets 1 & 2 digital twin](https://drive.google.com/file/d/1nXRN_VZ_DKGS33jtELwn6z8WR8dbQfKe/view?usp=sharing)

### Operating constraints

- Total vial volume must not exceed 80 mL.
- A selected component should use at least 10 mL in practice.
- T-200 must not exceed 50 mL.
- T-300 must not be used.
- Recipe changes require plant-manager guidance.

### Hypotheses

**H1 — A simple visual baseline is competitive.** Stable color and fill-level measurements will establish meaningful performance before a vision-language model is needed.

**H2 — Recipe history predicts carryover.** Previous/current recipe composition and dispense order will predict the first-vial color shift better than the current recipe alone.

**H3 — Multiview evidence improves explanation.** Linking tank, fill, and cap/seal frames will improve cause attribution even when anomaly detection itself changes little.

### Baseline ladder

1. **Synchronization audit:** measure missing frames, nearest-frame deltas, clock offset, and drift.
2. **Vial localization:** identify a stable region of interest during and after fill.
3. **Color/fill baseline:** compute robust Lab/HSV color and geometric fill measurements.
4. **Recipe reference:** use the last vial of each recipe as its clean-color reference.
5. **Transition model:** predict first-vial risk using prior/current recipe and dispense order.
6. **Multiview explanation:** attach aligned tank/fill/cap evidence to the decision.
7. **Synthetic augmentation:** test whether varied synthetic fill, lighting, and color improve unseen-recipe performance.

### Split and validation protocol

- Use leave-one-recipe-out evaluation across all ten recipes.
- Keep all frames from one vial/event in the same split.
- Split additional unlabeled/manual-review data by run or time window.
- Evaluate anomaly detection and cause attribution as separate tasks.
- Calibrate the color pipeline per camera without using held-out recipe outcomes.

### Scorecard

| Metric | Why it matters |
|---|---|
| Color anomaly recall | Finds discolored product |
| Fill anomaly recall | Finds under/overfill conditions |
| False alarms | Measures needless intervention |
| Cause-attribution accuracy | Tests whether the explanation is useful |
| Transition-risk precision/recall | Tests pre-fill warning ability |
| Unseen-recipe performance | Measures high-mix/low-volume generalization |
| Timestamp alignment error | Establishes trust in multiview evidence |

### Failure modes to attack early

- Auto-exposure or white balance shifts masquerade as recipe changes.
- Transparent plastic and background dominate average color.
- Adjacent frames leak across splits and inflate performance.
- First-vial carryover is treated as an isolated visual anomaly.
- Approximate twin color teaches an unrealistic mapping.
- Barcode presence or orientation becomes an accidental shortcut.

### Winning version

A transition-aware operator assistant that warns when a queued recipe is likely to be contaminated, then explains a detected anomaly using recipe history and synchronized visual evidence. The demo should separate “what happened,” “why it probably happened,” and “what the operator should verify.”

---

## Experiment 3 — Grasping and Sorting

### Manufacturing problem

Conventional fixed-motion arms work only when objects arrive predictably. The experiment asks teams to use multiple cameras and a digital twin to make a FANUC ER-4iA with a SCHUNK pneumatic two-finger gripper grasp and sort reliably despite variation.

The promised setup includes two overhead cameras, one wrist camera, a digital twin, camera/calibration samples, current-finger CAD, and supervised access to design and print replacement fingers.

At the time of the saved specification, some download and access details were still pending. Asset availability is therefore the first go/no-go test.

### Hypotheses

**H1 — Calibration dominates early performance.** Improving camera-to-robot calibration will yield more reliable picks than changing the learning model during the first baseline.

**H2 — Closed-loop recovery is the real differentiator.** Re-observation and safe retry will improve task completion more than optimizing one ideal open-loop grasp.

**H3 — Controlled randomization transfers.** Variation in position and orientation will transfer more reliably than aggressive randomization across every visual and physical property at once.

### Baseline ladder

1. Load the twin and render a short verified pick-and-place.
2. Reproduce the existing object/finger setup without learning.
3. Validate camera calibration and coordinate transforms.
4. Measure repeated picks across randomized positions.
5. Add object orientation variation.
6. Add grasp verification, re-observation, and retry.
7. Compare a small supervised physical-trial set with simulation.
8. Only then test new objects, clutter, learned policies, or finger geometry.

### Split and validation protocol

- Define development and held-out regions of the workspace.
- Pre-register position/orientation ranges and object sets.
- Run enough repeated trials to report uncertainty, not a single success video.
- Preserve identical success definitions in simulation and reality.
- Log collisions, safety stops, timeouts, and operator interventions.

### Scorecard

| Metric | Why it matters |
|---|---|
| Successful picks per attempt | Core reliability |
| Placement accuracy | Confirms useful manipulation |
| Cycle time and tail latency | Exposes slow failure paths |
| Recovery success | Measures closed-loop intelligence |
| Operator interventions | Indicates autonomy and usability |
| Performance on unseen poses/objects | Tests generalization |
| Sim-to-real gap | Quantifies transfer quality |

### Failure modes to attack early

- Incorrect coordinate transform produces consistent misses.
- Wrist-camera occlusion hides the grasp point.
- Pneumatic or friction behavior differs from simulation.
- Collision-free simulated motion lacks real clearance.
- A custom finger improves one object but destroys generality.
- The team loses build time waiting for physical access or missing files.

### Safety boundary

Do not command the physical arm, change controller configuration, or install printed fingers without the plant manager and assigned mentor. Review clearance and safe operation before every physical modification.

### Winning version

A repeatable robot demonstration that measures success over many trials and visibly recovers from a missed or uncertain grasp. The story should show calibration, perception, action, verification, retry, and the measured sim-to-real gap.

---

## Comparative experiment backlog

| Priority | Experiment | Smallest useful test | Decision unlocked |
|---:|---|---|---|
| P0 | All | Verify downloads, schemas, restrictions, and physical access | Can we select this dataset safely? |
| P0 | Vial | Render all fixture targets over real scenes | Is geometry metadata reliable? |
| P0 | Mixing | Align one event across all three cameras | Is multiview reasoning feasible? |
| P0 | Robot | Load the twin and render one verified motion | Are the robotics assets usable? |
| P1 | Vial | Golden-reference or tiny classifier baseline | How large is the real domain gap? |
| P1 | Mixing | Color/fill measurement on 30 labeled vials | Is a larger model justified? |
| P1 | Robot | Repeated simulated picks over position grid | What fails before real hardware? |
| P2 | Vial | Fixture fill-state reasoning | Does context reduce false alarms? |
| P2 | Mixing | Prior-recipe transition model | Can we predict carryover? |
| P2 | Robot | Re-observe and retry | Does closed-loop recovery add value? |

## Recommended team-plan entry

> **The A-Team proposes to begin with Dataset 1: Final Vial Inspection.** We will train an inspection model using labeled synthetic images from the digital twin and evaluate it on held-out real-line imagery. Our first baseline will classify each numbered fixture position and combine that prediction with the known fill sequence to distinguish missed placements from positions that are not yet expected to be occupied. We will measure recall per defect class, escaped defects, false alarms, fixture-level accuracy, and inference latency. We request the complete digital twin, fixture metadata, labeling-tool export schema, held-out real test data when available, and a supervised CSI session to validate the final system. We will retain Dataset 2 as the fallback if the vial assets or synthetic-to-real baseline are not viable during the initial gate.

## Immediate questions for organizers

1. Are the real samples strictly evaluation-only, and may they be used for threshold calibration?
2. Will a held-out real test set be released, and how will teams submit against it?
3. What order-complete or placement-count signals are available for Dataset 1?
4. Are camera exposure and white-balance settings fixed for Dataset 2?
5. Is recipe history/control metadata available live or only in the labeled export?
6. Are all Dataset 3 twin, calibration, camera, and finger-CAD assets now released?
7. What supervised hardware-test windows are available before October 28?
8. What latency or deployment hardware should teams target for the final demonstration?

## Source and confidence note

This entry is derived from the three October 2026 dataset-description pages supplied with the kickoff archive and the recorded kickoff materials. It describes the promised structure—not a completed audit of the raw assets. File counts, label balance, corrupt media, timestamp quality, and the real/synthetic domain gap must be measured after download.
