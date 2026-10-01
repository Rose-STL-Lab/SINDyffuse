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
| 3 — Polynomial cache | `scripts/build_lai_opensimad_polynomials.py` | one-time full-ROM muscle path fitting in isolated 100-frame processes |
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
Logs preserve run history across pod deletion; they do **not** checkpoint or resume computation.
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
