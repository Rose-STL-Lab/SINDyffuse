# SINDyffuse

Text-conditioned human motion diffusion with **SINDy** biomechanics targets and **Nimble/OpenSim** physics guidance.

HumanML3D joint trajectories are retargeted to the Rajagopal 2015 musculoskeletal model, cached as Nimble B3D files, and used to train:

1. **SINDy** — text → sparse coefficients for **120 targets** (40 L_bio + 80 muscle activations)  
2. **Activation surrogate** — fast `q` → 80 muscle activations (OpenSim labels at preprocess)  
3. **Diffusion** — text → motion with SINDy guidance (`loss_diff + lambda_sindy * loss_sindy`)  

## Setup

```bash
conda env create -f env/environment.yaml
conda activate sindyffuse
```

OpenSim and the Rajagopal `.osim` model come from the `opensim` and `nimblephysics` conda/pip packages (no bundled geometry in this repo).

Point at your HumanML3D checkout:

```bash
export HUMANML3D_ROOT=/path/to/HumanML3D   # optional; default: datasets/HumanML3D
```

## Data layout

```
datasets/HumanML3D/          # not committed (~20GB)
  new_joint_vecs/
  texts/
  train.txt, val.txt, test.txt
  nimble_b3d/                # canonical B3D cache (IK / static / Moco — one folder)
    {motion_id}.b3d
    Mean.npy, Std.npy
```

## Pipeline

Run entry points from the repo root:

```bash
cd /path/to/SINDyffuse
```

### 1. Preprocess (four-job MinT-style pipeline)

Production preprocessing is **four sequential jobs** sharing the same Python scripts for local runs and Kubernetes indexed jobs:

| Job | Script | Purpose |
|-----|--------|---------|
| 1 — IK | `scripts/preprocess_ik.py` | joints → `q`, SINDy/guidance features, zero activations |
| 2 — Path fit | `scripts/fit_rajagopal_function_paths.py` | one-time `FunctionBasedPathSet.xml` from 200 stratified IK B3D samples |
| 3 — MocoTrack | `scripts/preprocess_moco.py` | muscle activations + GRF + validity mask (reads IK B3D, no IK redo) |
| 4 — Norm | `scripts/compute_normalization.py` | merge moco manifests → `Mean.npy` / `Std.npy` |

```bash
python scripts/preprocess_ik.py --max_motions 5
python scripts/fit_rajagopal_function_paths.py --sample_motions 200
python scripts/preprocess_moco.py --max_motions 5
python scripts/compute_normalization.py --num_shards 1 --wait
```

`scripts/preprocess_nimble.py` remains as a **deprecated** local wrapper (IK then Moco). Prefer the split scripts above.

**IK quality gates (Job 1 only):** fixed MinT mint-analysis translational threshold — max/mean mapped joint position error **≤ 0.02 m**, all frames must converge (`success_ratio = 1`), plus structural checks (valid `q`, ≥ 2 frames). Failed motions are `ik_failed` in the manifest; Moco skips them via prior status only (no FK re-filter at Moco).

**MinT-aligned Moco defaults:** `mesh_interval=0.02`, `convergence_tolerance=1e-3`, `max_iterations=3000`, `adaptive_mesh=False`. Segment success is **Ipopt success ∧ parsed activations** only (no reserve QC). Manifest statuses: `ik_ok` / `ik_failed` (Job 1), `ok` / `moco_failed` / `moco_skipped` (Job 3).

By default, Moco K8s pods run **one segment at a time with all CPUs** (`MOCO_PARALLEL_SEGMENTS=1`). Optional `--moco_parallel_segments 6` on fat local nodes after pilot.

Each `.b3d` stores generalized coordinates plus custom channels: `guidance_features`, `sindy_features`, `muscle_activations` `[80, T]`, and (MocoTrack) `sim_grf` `[18, T]` plus `muscle_activation_mask` `[1, T]`.

At **20 fps**, segmented Moco uses **28-frame cores**, **3-frame buffers**, and **34-frame solve windows** (1.4 s / 0.14 s MinT timing).

**MocoTrack** — MinT-style segmented trajectory optimization with foot contact: ground offset → 1.4 s Moco windows → seam stitch. Reference coordinates are low-pass filtered at **6 Hz**. Failed segments leave **NaN gaps**; the validity mask marks good frames. Training uses gap-aware window indexing (`nimble/gap_utils.py`).

**Static optimization** (`static_optimization`) — still available via deprecated `preprocess_nimble.py --activation_method static_optimization`.

OpenSim console output is **hidden by default** (`--opensim_log_level Off`).

Useful Moco flags: `--moco_core_duration_s`, `--moco_buffer_duration_s`, `--moco_stitch_blend_s`, `--moco_reference_lowpass_hz`, `--moco_states_speed_tracking_weight`, `--moco_no_reference_lowpass`, `--moco_mesh_interval`, `--moco_parallel_segments`, `--opensim_log_level`.

**Kubernetes:**

```bash
kubectl delete job sindyffuse-preprocess-moco-track -n YOUR_NAMESPACE   # before redeploy
kubectl apply -k deploy/jobs/preprocess-nimble/ik
kubectl apply -k deploy/jobs/fit-function-paths
kubectl apply -k deploy/jobs/preprocess-nimble/moco-track
kubectl apply -k deploy/jobs/preprocess-nimble/normalization
```

Local sharded test:

```bash
python scripts/preprocess_ik.py --max_motions 8 --num_shards 4 --shard_index 0 --skip_normalization
python scripts/preprocess_moco.py --max_motions 8 --num_shards 4 --shard_index 0 --skip_normalization --num_workers 0
python scripts/compute_normalization.py --num_shards 4 --wait
```

After upgrading the B3D schema (e.g. L_bio v2 with 40 `guidance_features` rows), **re-run preprocess** without `--skip_existing` on old caches.

### 2. Train SINDy

Requires B3D cache with **muscle activations** (preprocess with `moco_track` or `static_optimization`, not `none`).

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

Config: `configs/train_diffusion.json`. With `guidance=sindy`, loss is **diffusion denoising + SINDy consistency** only (no Nimble term). SINDy guidance compares `Θ(q)·Ξ(text)` to `actual(q)` where bio channels use FK physics and muscle channels use the **activation surrogate** at inference time. Set `train.sindy_checkpoint_dir` and `train.surrogate_checkpoint_dir`.

### 5. Generate motion

```bash
python scripts/generate_motion.py --checkpoint results/diffusion/latest.pt \
  --caption "a person walks forward" --out_npz out.npz \
  --guidance sindy \
  --sindy_checkpoint_dir results/sindy/latest \
  --surrogate_checkpoint_dir results/activation_surrogate/latest
```

### 6. Evaluate

Requires generated motions as NPZ files (`motion` array `[T, 37]`) under `--generations_dir`, plus HumanML3D B3D cache for biomechanical metrics.

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
./deploy/scripts/run-preprocess-nimble.sh static-optimization
# Or normalization alone after workers finished:
kubectl apply -k deploy/jobs/preprocess-nimble/normalization
kubectl apply -k deploy/jobs/train-sindy
kubectl apply -k deploy/jobs/train-surrogate
kubectl apply -k deploy/jobs/train-diffusion/nimble

# Interactive dev shell on the cluster
kubectl apply -k deploy/dev
kubectl exec -it sindyffuse-dev -- bash -l
```

See [deploy/README.md](deploy/README.md) for image build, storage setup, and the full job list.

**Container image:** Build locally with `env/Dockerfile` (`docker build -f env/Dockerfile .`). No public registry URL is provided for review.

## Project layout

| Path | Role |
|------|------|
| `scripts/preprocess_ik.py` | Job 1: HumanML3D → IK B3D cache |
| `scripts/preprocess_moco.py` | Job 3: MocoTrack on IK B3D cache |
| `scripts/fit_rajagopal_function_paths.py` | Job 2: function-based muscle paths |
| `scripts/preprocess_nimble.py` | Deprecated wrapper (IK + Moco) |
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
| `nimble/` | IK, B3D I/O, OpenSim muscle activation, Rajagopal guidance |
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
python scripts/preprocess_nimble.py --max_motions 1 --opensim_log_level Warn
```

- Re-run `scripts/preprocess_nimble.py` after upgrading B3D schema (e.g. adding `muscle_activations`).  
- If Ctrl+C does not stop Moco: `pkill -9 -f "python scripts/preprocess_nimble.py"`.
