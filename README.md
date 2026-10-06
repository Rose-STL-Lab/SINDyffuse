# SINDyffuse

Text-conditioned human motion diffusion with **SINDy** biomechanics targets and optional **OpenSim** CPU guidance (paper baselines).

HumanML3D joint trajectories are retargeted to **LaiUhlrich2022** (OpenSim IK + OpenSimAD), cached as per-motion **NPZ** files, and used to train:

1. **SINDy** — text → sparse coefficients for **120 targets** (40 L_bio + 80 muscle activations)  
2. **Activation surrogate** — fast `q` → 80 muscle activations (OpenSimAD labels at preprocess)  
3. **Diffusion** — text → motion with SINDy guidance by default (`loss_diff + lambda_sindy * loss_sindy`); `guidance=opensim` is an optional slow CPU baseline  

## Setup

```bash
conda env create -f env/environment.yaml
conda activate sindyffuse
```

OpenSim comes from the `opensim` conda package. The Lai model lives under `models/lai_uhlrich/`. Optional Geometry meshes: local folder or `LAI_GEOMETRY_SRC` (not required for IK / OpenSimAD).

Point at your HumanML3D checkout:

```bash
export HUMANML3D_ROOT=/path/to/HumanML3D   # optional; default: datasets/HumanML3D
```

## Data layout

```
datasets/HumanML3D/          # not committed (~20GB)
  new_joints/ | joints/
  texts/
  train.txt, val.txt, test.txt
  lai_cache/                 # per-motion NPZ (IK + OpenSimAD)
    {motion_id}.npz          # q [T,31], activations, mask, GRF, features
    Mean.npy, Std.npy        # q mean/std [31]
    cache_meta.json
```

## Pipeline

Run entry points from the repo root:

```bash
cd /path/to/SINDyffuse
```

### 1. Preprocess (IK → OpenSimAD → norm)

| Job | Script | Purpose |
|-----|--------|---------|
| 1 — IK | `scripts/preprocess_ik.py` | joints → `lai_cache/{id}.npz` with `q` (+ placeholders) |
| 2 — Compiled OpenSimAD ext | `scripts/build_lai_opensimad_ext.py` | one-shot AD model + validated `F.so` |
| 3 — Polynomial cache | `scripts/build_lai_opensimad_polynomials.py` | prepare → indexed slice extraction → aggregate/full-ROM fitting (single-process mode also available) |
| 4 — Canary | `scripts/run_opensimad_canary.py` | one MinT-sized solve before worker fan-out |
| 5 — Activations | `scripts/preprocess_moco.py` | MinT/OpenSimAD → activations + GRF + validity mask (patches NPZ) |
| 6 — Norm | `scripts/compute_normalization.py` | merge manifests → `Mean.npy` / `Std.npy` |

```bash
python scripts/preprocess_ik.py --max_motions 5
python scripts/build_lai_opensimad_ext.py
python scripts/build_lai_opensimad_polynomials.py --num_threads 1
python scripts/run_opensimad_canary.py
python scripts/preprocess_moco.py --max_motions 5 --activation_method opensimad
python scripts/compute_normalization.py --num_shards 1 --wait
```

Path-fit (`fit_rajagopal_function_paths.py`) is **not** required on this branch (OpenSimAD uses polynomial MT paths).

**Kubernetes (full preprocess pipeline):**

```bash
./deploy/scripts/preprocess-dataset-orchestrate.sh full YOUR_NAMESPACE
```

Or run stages individually:

```bash
./deploy/scripts/preprocess-dataset-orchestrate.sh ik YOUR_NAMESPACE
./deploy/scripts/preprocess-dataset-orchestrate.sh opensimad YOUR_NAMESPACE
```

Optional direct Job apply (without local orchestrator):

```bash
kubectl apply -k deploy/jobs/preprocess-dataset/inverse_kinematics -n YOUR_NAMESPACE
```

**IK quality gates (Job 1):** structural checks only — valid `q`, ≥ 2 frames, and all frames must converge (`success_ratio = 1`). HumanML3D joint-position fit stats are recorded for diagnostics but are **not** used to reject motions. Failed IK motions are `ik_failed` in the manifest; activation skips them via prior status only.

**Activation gates (Job 5, MinT gap policy):** a motion is `ok` if **≥1** OpenSimAD segment succeeds. Failed segments leave **NaN gaps** and `muscle_activation_mask=0`. Whole-motion coordinate RMSE is diagnostic only (no longer hard-fails the clip). Manifest statuses: `ik_ok` / `ik_failed` (Job 1), `ok` / `moco_failed` / `moco_skipped` (Job 5).

For memory safety, activation pods run **one segment at a time**, and each segment runs in a fresh spawned process. The cluster still runs up to 180 indexed pods in parallel. Settings remain MinT-aligned: **2500** Ipopt iterations, mesh interval **0.02 s** (50 collocation points/s), 1.4-second cores, and 0.14-second buffers.

Workers require the prebuilt `F.so` and polynomial cache. They never fall back to the high-memory `F.py` expression graph or perform polynomial MuscleAnalysis lazily. A segment outside the precomputed full model range becomes a failed/gap segment rather than launching a worker-local fit.

**Polynomial-build diagnostics:** Kubernetes attempts write separate console `.log` files and
structured `.jsonl` files to `/mnt/SINDyffuse/logs` on the PVC (not `/scratch`). The JSONL path
is printed at startup in the console log. Records include UTC timestamps, PID/pod identity,
process RSS, available cgroup memory counters, a 10-second memory heartbeat, stage durations,
chunk ranges, per-frame coordinates in degrees, and polynomial fitting order/error diagnostics.
Python exceptions include tracebacks. After an OOM/SIGKILL there may be no final failure record;
look for the last `stage_start`/`frame_start` without its corresponding completion and inspect
nearby memory records. Kubernetes termination status is still needed to establish the kill reason.
The distributed path also checkpoints completed slices on the PVC; the legacy `single` mode
only preserves run history and does not checkpoint computation.
For local builds, JSONL logs default to the repository's `logs` directory; use `--log_dir` to
select a persistent location and `--memory_log_interval` to change heartbeat frequency.

Instrumentation does not change samples, full-ROM bounds, muscle sets, polynomial order
(3–9), the existing 0.0015 fitting threshold, or the least-squares algorithm. The MinT
[paper, Appendix A.3](https://arxiv.org/html/2411.00128v1#A1.SS3) specifies the downstream
50-point/s mesh, 1e-3 tolerance, 2500 iterations and 1.4 s cores/0.14 s buffers; those remain
unchanged. MinT's [public code](https://github.com/simplexsigil/MusclesInTime) provides dataset
utilities and muscle definitions, not the simulation-generation pipeline. Polynomial fitting
here follows the vendored [OpenCap implementation](https://github.com/stanfordnmbl/opencap-processing/blob/main/UtilsDynamicSimulations/OpenSimAD/polynomialsOpenSimAD.py),
rather than claiming verified exact parity with MinT's unpublished generation code.

### Distributed polynomial cache build

**Activation pilot after an ordinary nonconverged canary:** once artifacts validate and
the solver runs without systemic numerical errors, a bounded multi-motion pilot can test
coverage without accepting any failed solve:

```bash
kubectl apply -n ai-md -k /Users/nick.king/Documents/git_repos/SINDyffuse/deploy/jobs/preprocess-dataset/moco-track-pilot
```

The pilot runs five concurrent indexed pods over the first 25 prepared motion tasks.
It uses the same solver/ROM checks and does not relax convergence. Prepare the full task
manifest first using `prepare-moco-tasks`. Do not run pilot and production workers
concurrently: they write the same NPZ/task outcomes.
Job completion does not imply every segment succeeded; inspect per-motion manifests and
usable surrogate windows before applying the full `moco-track` Job. A systemic code,
artifact, OOM or nonfinite error still requires diagnosis rather than blind fan-out.

### One-motion-per-pod activation scheduling

`moco-track` now has 29,228 indexed tasks, up to 180 concurrent pods, and exactly one
motion per pod. Each pod retains bounded segment multiprocessing: up to eight isolated
segment processes, four solver threads each, within the existing 32 CPU/32 GiB allocation.
No solver tolerances or iteration limits change. Pending indices are queued by Kubernetes.
CPU and memory settings remain adjustable; keep threads × segment workers within the CPU
budget and benchmark peak memory. Long motions launch only the number of segments needed.

Before deployment, **stop the old sharded Job and wait for its pods to terminate**, then
sync the checkout. Do not update the mounted code while the old fleet is processing.
Preparation freezes the ordered IDs, solver configuration, code/artifact provenance, IK
status records and dataset root under `datasets/HumanML3D/motion-tasks/default/tasks.json`.
It refuses a changed assignment/configuration; select a new `MOTION_TASK_DIR` in preparation,
workers and normalization if those inputs change. `MOTION_TASK_COUNT` and Job completions
must equal the number of selected IDs (29,228 by default); changing `parallelism` alone does
not change task identity. `MAX_MOTIONS` can limit a prepared selection but must be matched
by those counts. Preparation and normalization preserve legacy manifests for audit.

With already rebuilt model/polynomial artifacts, deploy in this order:

```bash
kubectl apply -n ai-md -k /Users/nick.king/Documents/git_repos/SINDyffuse/deploy/jobs/preprocess-dataset/prepare-moco-tasks
kubectl wait -n ai-md --for=condition=complete job/sindyffuse-prepare-moco-tasks --timeout=2h
kubectl apply -n ai-md -k /Users/nick.king/Documents/git_repos/SINDyffuse/deploy/jobs/preprocess-dataset/moco-track
# After the indexed Job succeeds:
kubectl apply -n ai-md -k /Users/nick.king/Documents/git_repos/SINDyffuse/deploy/jobs/preprocess-dataset/normalization
```

Per-motion outcomes are atomically written under `motion-tasks/default/outcomes`; per-index
locks prevent duplicate replacement pods writing the same motion concurrently. NPZ files
are compressed to a private temporary file, flushed, validated and atomically replaced,
so interruptions do not truncate an existing readable cache. Existing finite activations
are imported without solving and explicitly marked as imported (legacy artifacts have no
per-task provenance). New results record their checksum and task-set identity. Retries reuse
validated successes. Ordinary ROM rejection/nonconvergence is a completed dataset task
with invalid labels—not a Kubernetes failure. Unclassified processing/data errors fail
the index, which has two retries. Normalization requires all correctly identified terminal
outcomes and rejects unresolved errors; a complete Job does not mean every motion converged.
To intentionally retry deterministic dataset failures, select a new task directory rather
than rewriting the immutable outcome history. Per-segment scratch is still disposable;
pod loss can repeat one motion, but never an entire fixed shard.

Headless activation workers explicitly request machine-readable kinematics/activation
and resultant-GRF exports (`writeMachineReadable=True`, `writeGUI=False`). Solver success
alone is not sufficient: the wrapper must find and parse the exported labels. This fixes
the historical mismatch where successful solves were reported as missing-output failures.
The wrapper consumes `GRF_resultant_*.mot`, not the per-contact-sphere file. Model/F.so/
polynomial rebuilds are not required for this export-only repair. Existing converged
`w_opt_0.npy` and `stats_0.npy` can be reanalyzed with `solveProblem=False` if their complete
scratch session survives; scratch is not a persistent recovery checkpoint and is lost
when a pod is deleted. Restarted workers otherwise solve again. Avoid changing the mounted
checkout while old workers are still active; deploy the fix before restarting them.

**Label processing (`overlap_crossfade_tracking_grf_v1`):** full solve-window predictions
are retained until assembly. Neighboring successful segments are linearly crossfaded
inside their common buffered interval (default blend radius 0.14 s); a failed core remains
NaN even if a neighbor has predictions there. This implements the seam-smoothing intent
described in MinT Appendix A.3, not a verified copy of their unpublished algorithm.
Blending is postprocessing, not a claim that blended signals satisfy the original dynamics
exactly. Unblended interior frames are unchanged, and gaps are never interpolated away.

Per-segment pose tracking RMSE/max errors are computed from `optimaltrajectories.npy`;
rotations are converted from radians to degrees, translations remain metres. The
reference is OpenCap's filtered target including optimized pelvis offset, compared at
common mesh timestamps. Sample-count-weighted pooled errors and MinT-analysis flags
(rotational RMSE <5 degrees and translational RMSE <0.02 m) are persisted diagnostically,
not used to reject converged segments. No new activation/saturation/rate cutoff is added.
An empty metric set reports unavailable values, not zero error. Tracking against original
unfiltered IK is distinct and is not implied by these flags.

NPZ files retain `activation_diagnostics_json` with processing version, segment solver
status/iterations, tracking metrics and torque convention. Resultant torque channels are
free moments at the COP, not full ground-reaction moments about the origin. GRF trust
is 1 only where all 12 force/free-moment channels are finite; missing endpoint forces
remain NaN without extrapolation. Activation validity is separate from GRF validity.
Old caches are readable but do not gain this processing/metadata retroactively. Imported
task outcomes mark them `legacy_unreported`; to upgrade, retain complete scratch outputs
for reanalysis or recompute activations with skip-existing disabled and a new task directory.
No model, compiled external-function or polynomial rebuild is required for these changes.

**Surrogate windows:** `window_size=64` is maximum context, not a minimum usable run.
`min_window_size=1` retains shorter contiguous valid runs, including isolated frames.
Partial motions are no longer rejected based on whole-motion NaN percentage. Batches pad
short windows; attention, activation loss and validation metrics ignore padding. Temporal
loss only uses adjacent valid frames, and is absent for a one-frame window. Invalid labels
are never interpolated, zero-filled as targets, or joined across a gap. Minimum context
is adjustable for a learning experiment, not a new MinT biomechanical quality gate.
Long valid runs use sliding windows with tail coverage; coverage counters and the window
policy are stored in logs/checkpoints. Legacy relaxed validity flags cannot enable NaN
targets. Changing minimum context changes training data coverage, so record it with results.

**Required rebuild after the unit/model-conversion repair:** polynomial IK tables now
declare `inDegrees=yes`, and NumPy spline coefficients are reversed for OpenSim's
descending-power convention. The old `default` build and caches derived from it are
invalid and must not be reused. All three manifests now point at a new
`polynomial-builds/units-model-v2` directory. After syncing this code, rerun `build-ext`
(forced model/F.so rebuild), then `build-polynomials` (prepare → all slices → finalize),
then `canary`. Existing IK NPZ files need not be regenerated for these AD-stage fixes.
Cache metadata is versioned so old artifacts fail worker preflight.

**Quality policy and sources:** [MinT Appendix A.3](https://arxiv.org/html/2411.00128v1)
specifies 50 collocation points/s, 1e-3 solver tolerance, 2500 iterations, and discarding
nonconverged segments; those settings remain unchanged. Its public
[analysis code](https://github.com/simplexsigil/mint-analysis) reports tracking/GRF/
activation/KAM checks, while the dataset README describes that metadata as optional
for filtering. These flags are not added as hard cache/dataset rejection gates.
Neither source supplies muscle-length/moment-arm or spline-fit rejection cutoffs, so
none are invented here. Unit and coefficient evaluation assertions are software
consistency checks. Spline approximation errors, raw geometry ranges, and true fitting
RMSE are logged diagnostically. The inherited OpenCap fitting criterion and maximum-order
acceptance remain unchanged (the 0.0015 fit threshold is OpenCap's, not MinT's solver
tolerance). Nonfinite initial NLP values fail with constraint descriptions; this checks
that the numerical problem is defined, not a new biomechanical threshold.

The OpenSimAD canary automatically selects a finite diagnostic window that remains within
the published polynomial ROM after the solver's 6 Hz filter and mesh interpolation. It
does not clip motion, expand bounds, or permit worker-side fitting. Among eligible windows
it retains the highest-variability selection rule. `--max_candidates` controls the sorted
scan budget (default 128); an explicit out-of-domain `--motion_id` fails with coordinate
ranges instead of silently selecting another motion. Canary console/JSONL logs persist
under `/mnt/SINDyffuse/logs`, and failed solver reports now include tracebacks.

Run the three stages in sequence using the local orchestrator:

```bash
./deploy/scripts/preprocess-dataset-orchestrate.sh build-polynomials YOUR_NAMESPACE
```

This recreates the stage Jobs (including any running Job with the same name), but retains
validated slice results. Do not run it concurrently with the old single-pod build or another
cache publisher. Ensure this checkout is synchronized to `/mnt/SINDyffuse` before starting.

1. `prepare-opensimad-polynomials`: publishes immutable model/samples/manifest inputs.
2. `build-opensimad-polynomials`: **200 indexed tasks**, **100 concurrent pods**, one
   10-frame slice per task. Workers retain the existing 256 GiB/1 CPU requests and limits.
3. `finalize-opensimad-polynomials`: validates every slice, assembles original frame order,
   fits both sides with the existing algorithm, and publishes the final cache.

All three stages use the same `POLYNOMIAL_BUILD_DIR`, defaulting to
`/mnt/SINDyffuse/models/lai_uhlrich/opensimad/polynomial-builds/units-model-v2`. Successful slices
are atomically published under `chunks/`; retries reuse them after validating metadata,
shapes, finite float64 values, exact input coordinates, and result checksums. Missing slices
block finalization. Invalid slice files are recomputed, not silently dropped.

**Adjustable configuration (edit manifests before applying):**

| Setting | Location | Default |
|---|---|---|
| Maximum concurrent pods | extraction `job.yaml`: `spec.parallelism` | 100 |
| Frames per slice | preparation env: `POLYNOMIAL_CHUNK_FRAMES` | 10 |
| Total indexed tasks | extraction `spec.completions`, and `POLYNOMIAL_EXPECTED_CHUNKS` in all three Jobs | 200 |
| Persistent build directory | `POLYNOMIAL_BUILD_DIR` in all three Jobs | shared path above |

Total tasks must equal `ceil(sample_count / chunk_frames)`; preparation and workers reject
mismatches. Concurrency is independent of the numerical build identity. Changes to chunk
size, model, sample generation, extraction/fitting code or runtime require a **new build
directory**. The manifest records seed, bounds, hashes, runtime versions and fitting settings;
existing inputs/results are never force-deleted. Keep the mounted checkout and environment
unchanged for the duration of a build. Use the same directory to resume an unchanged build.

For manual deployment, apply/wait for preparation, then extraction, then finalization;
applying the extraction manifest alone no longer builds a complete cache. Local equivalents:

```bash
python scripts/build_lai_opensimad_polynomials.py --mode prepare --build_dir /persistent/build --chunk_frames 10 --expected_chunks 200
python scripts/build_lai_opensimad_polynomials.py --mode extract --build_dir /persistent/build --chunk_index 0 --expected_chunks 200
# Run every index 0..199, then:
python scripts/build_lai_opensimad_polynomials.py --mode finalize --build_dir /persistent/build --expected_chunks 200
```

The original command remains available with `--mode single` (the default). Orchestration
wait budgets are configurable through `POLYNOMIAL_PREPARE_TIMEOUT` (2h),
`POLYNOMIAL_EXTRACT_TIMEOUT` (48h), and `POLYNOMIAL_FINALIZE_TIMEOUT` (12h); these are
local wait timeouts, not Kubernetes runtime deadlines. Preparation/finalization request
16 GiB each; adjust their manifests if measured fitting peaks require more memory.

**Finalizer-only recovery from the legacy five-coordinate basis bug:** the fitter and
runtime evaluator now support muscles spanning six or more coordinates without dropping
moment arms. The finalizer manifest enables `--allow_fitting_upgrade`, which permits only
the known predecessor hashes for this fitting-only repair. All input, runtime, extraction
code and slice checks remain enforced. It does not rewrite the prepared manifest or change
the slices. Published metadata records the original build ID and new finalization code.
For this specific failure, sync the fixed code to `/mnt/SINDyffuse`, recreate only the
`sindyffuse-finalize-opensimad-polynomials` Job, and apply its manifest; do not rerun the
full polynomial orchestrator (preparation deliberately rejects changed build code).

Each `{id}.npz` stores generalized coordinates `q` `[T, 31]` plus `muscle_activations` `[T, 80]`, `muscle_activation_mask` `[T]`, `sim_grf` `[T, 18]`, and SINDy feature rows.

At **20 fps**, segmented OpenSimAD uses **28-frame cores**, **3-frame buffers**, and **34-frame solve windows** (1.4 s core / 0.14 s buffer).

**OpenSimAD** — segmented trajectory optimization with foot contact: ground offset → 1.4 s windows → seam stitch (MinT). Reference coordinates are low-pass filtered at **6 Hz**. Failed segments leave **NaN gaps**; the validity mask marks good frames. Training uses gap-aware window indexing (`nimble/gap_utils.py`).

OpenSim console output is **hidden by default** (`--opensim_log_level Off`).

Useful flags: `--activation_method`, `--moco_core_duration_s`, `--moco_buffer_duration_s`, `--moco_stitch_blend_s`, `--moco_reference_lowpass_hz`, `--moco_mesh_interval`, `--moco_parallel_segments`, `--opensim_log_level`.

**Kubernetes (manual stage apply):**

```bash
kubectl delete job sindyffuse-preprocess-moco-track -n YOUR_NAMESPACE   # before redeploy
kubectl apply -k deploy/jobs/preprocess-dataset/build-opensimad-ext -n YOUR_NAMESPACE
kubectl apply -k deploy/jobs/preprocess-dataset/build-opensimad-polynomials -n YOUR_NAMESPACE
kubectl apply -k deploy/jobs/preprocess-dataset/opensimad-canary -n YOUR_NAMESPACE
kubectl apply -k deploy/jobs/preprocess-dataset/moco-track -n YOUR_NAMESPACE
```

Local sharded test:

```bash
python scripts/preprocess_ik.py --max_motions 8 --num_shards 4 --shard_index 0 --skip_normalization
python scripts/preprocess_moco.py --max_motions 8 --num_shards 4 --shard_index 0 --skip_normalization --num_workers 0
python scripts/compute_normalization.py --num_shards 4 --wait
```

After upgrading the lai_cache NPZ schema, **re-run preprocess** without `--skip_existing` on old caches.

### 2. Train SINDy

Requires `lai_cache/` with **OpenSimAD muscle activations** (`scripts/preprocess_moco.py`).

```bash
python scripts/train_sindy.py --output results/sindy
```

Config: `configs/train_sindy.json` (2000 epochs, batch 64, lr 1e-3; lowest validation MSE checkpoint). Joint model predicts **120 channels** (40 L_bio + 80 muscles) from text-conditioned sparse `Ξ(text)`.

### 3. Train activation surrogate

```bash
python scripts/train_surrogate.py --config configs/train_surrogate.json --output results/activation_surrogate
```

Config: `configs/train_surrogate.json` (500 epochs, batch 32, lr 1e-3; lowest validation L1 checkpoint). Temporal transformer architecture; L1 plus `lambda_temporal=0.15`.

### 4. Train diffusion

```bash
python scripts/train_diffusion.py --config configs/train_diffusion.json --out_dir results/diffusion
```

Config: `configs/train_diffusion.json`. With `guidance=sindy`, loss is **diffusion denoising + SINDy consistency** (default). Optional `guidance=opensim` is a slow OpenSim FK soft-constraint baseline. SINDy guidance compares `Θ(q)·Ξ(text)` to `actual(q)` where bio channels use OpenSim keypoints and muscle channels use the **activation surrogate**. Set `train.sindy_checkpoint_dir` and `train.surrogate_checkpoint_dir`.

### 5. Generate motion

```bash
python scripts/generate_motion.py --checkpoint results/diffusion/latest.pt \
  --caption "a person walks forward" --out_npz out.npz \
  --guidance sindy \
  --sindy_checkpoint_dir results/sindy/latest \
  --surrogate_checkpoint_dir results/activation_surrogate/latest
```

### 6. Evaluate

Requires generated motions as NPZ files (`motion` array `[T, 31]`) under `--generations_dir`, plus HumanML3D `lai_cache/` for biomechanical metrics.

```bash
python scripts/evaluate_motion.py \
  --generations_dir results/eval/generations \
  --data_root /path/to/HumanML3D \
  --split test \
  --out_json results/eval/metrics.json
```

For text-alignment metrics (R-Precision, FID, MM-Dist, Diversity), provide precomputed embeddings from the standard HumanML3D/T2M evaluator:

```bash
python scripts/evaluate_motion.py \
  --generations_dir results/eval/generations \
  --data_root /path/to/HumanML3D \
  --motion_embeddings /path/to/gen_emb.npy \
  --text_embeddings /path/to/text_emb.npy \
  --reference_motion_embeddings /path/to/ref_emb.npy \
  --out_json results/eval/metrics.json
```

Config: `configs/evaluate.json` (32 samples per caption, 1000 bootstrap replicates).

## Kubernetes

Job manifests live under `deploy/`. Configure your image and PVC in `deploy/components/cluster-config/`, then apply individual jobs:

```bash
./deploy/scripts/preprocess-dataset-orchestrate.sh full YOUR_NAMESPACE
# Or individual stages:
kubectl apply -k deploy/jobs/preprocess-dataset/inverse_kinematics -n YOUR_NAMESPACE
kubectl apply -k deploy/jobs/train-sindy -n YOUR_NAMESPACE
kubectl apply -k deploy/jobs/train-surrogate -n YOUR_NAMESPACE
kubectl apply -k deploy/jobs/train-diffusion/sindy -n YOUR_NAMESPACE
kubectl apply -k deploy/jobs/train-diffusion/opensim -n YOUR_NAMESPACE  # optional CPU baseline

# Interactive dev shell on the cluster
kubectl apply -k deploy/dev
kubectl exec -it sindyffuse-dev -- bash -l
```

See [deploy/README.md](deploy/README.md) for image build, storage setup, and the full job list.

**Container image:** Build locally with `env/Dockerfile` (`docker build -f env/Dockerfile .`). No public registry URL is provided for review.

## Project layout

| Path | Role |
|------|------|
| `scripts/preprocess_ik.py` | Job 1: HumanML3D → `lai_cache/` NPZ |
| `scripts/build_lai_opensimad_ext.py` | One-shot compiled OpenSimAD `F.so` build |
| `scripts/build_lai_opensimad_polynomials.py` | One-shot muscle polynomial cache build |
| `scripts/run_opensimad_canary.py` | Validate one MinT-sized solve before fan-out |
| `scripts/preprocess_moco.py` | Job 5: OpenSimAD activations → patch NPZ |
| `scripts/compute_normalization.py` | Merge shard manifests; compute `Mean.npy` / `Std.npy` |
| `scripts/train_sindy.py` | Train SINDy text→Xi model |
| `scripts/train_surrogate.py` | Train q→activation surrogate |
| `scripts/train_diffusion.py` | Train text-conditioned diffusion |
| `scripts/generate_motion.py` | Sample motion from trained diffusion |
| `scripts/evaluate_motion.py` | HumanML3D evaluation metrics |
| `eval/` | Metric computation and aggregation |
| `env/environment.yaml` | Conda environment |
| `env/Dockerfile` | Container image (local build) |
| `deploy/` | Kubernetes job manifests (see `deploy/README.md`) |
| `nimble/` | OpenSim IK / OpenSimAD helpers (no nimblephysics) |
| `datasets/lai_cache.py` | Per-motion NPZ schema + I/O |
| `surrogate/` | Differentiable activation surrogate (ML) |
| `sindy/` | SINDy library, dataset, training |
| `diffusion/` | Text-conditioned motion diffusion |
| `datasets/` | HumanML3D loaders (Python only; data is local) |

## Tests

```bash
conda activate sindyffuse
cd /path/to/SINDyffuse
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

OpenSim-backed tests require the `sindyffuse` conda env.

## Troubleshooting

```bash
python scripts/preprocess_ik.py --max_motions 1 --opensim_log_level Warn
python scripts/preprocess_moco.py --max_motions 1 --opensim_log_level Warn
```

- Re-run preprocess after upgrading the NPZ schema (e.g. adding `muscle_activations`).  
- Run the external-function, polynomial-cache, and canary stages before activation workers. Workers intentionally fail preflight when artifacts are missing or stale.
- If Ctrl+C does not stop activation workers: `pkill -9 -f "python scripts/preprocess_moco.py"`.
