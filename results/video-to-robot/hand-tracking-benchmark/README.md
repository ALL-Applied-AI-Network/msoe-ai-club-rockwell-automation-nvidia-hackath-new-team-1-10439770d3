# Local hand-tracking benchmark

This folder is an isolated, reproducible comparison of hand-tracking methods on
six retained assembly-demonstration videos. Supply their directory explicitly.

Run the OpenCV benchmark:

```powershell
python benchmark_hand_tracking.py --input-dir <path-to-demonstration-videos>
```

Also run the locally cached SAM2.1 Hiera Small video model (CPU-only and slower):

```powershell
python benchmark_hand_tracking.py --input-dir <path-to-demonstration-videos> --with-sam2
```

To preserve completed OpenCV results and run only the cached SAM2 test:

```powershell
python benchmark_hand_tracking.py --input-dir <path-to-demonstration-videos> --sam2-only
```

Outputs:

- `REPORT.md`: quantitative results, visual QA failures, and deployment recommendation.
- `inventory.json`: media and installed-package inventory.
- `benchmark_results.json` / `.csv`: direct runtime and temporal diagnostics.
- `annotated/*_h264.mp4`: reviewable comparison videos.
- `qa/`: source contact sheets and later visual QA frames.

The skin-overlap columns are consistency checks against a classical skin proxy;
they are **not labeled accuracy**.  Visual review and task-specific annotated
ground truth are still required before claiming production performance.
