# SINDyffuse

Text-conditioned human motion diffusion with **SINDy** biomechanics targets and **Nimble/OpenSim** physics guidance.

HumanML3D joint trajectories are retargeted to the Rajagopal 2015 musculoskeletal model, cached as Nimble B3D files, and used to train:

1. **SINDy** — text → sparse coefficients for **103 targets** (23 L_bio + 80 muscle activations)  
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

### 1. Preprocess (IK + OpenSim MocoTrack + feature cache)

```bash
python scripts/preprocess_nimble.py --max_motions 1
# Terminal: tqdm + errors only; verbose details in logs/preprocess_nimble_<timestamp>.log
# (--log_dir, --no_run_log on all main scripts)
```

Three muscle-activation modes via `--activation_method`:

| Method | CLI | Speed | Fidelity |
|--------|-----|-------|----------|
| Skip | `none` or `--skip_muscle_activation` | Fastest (IK only) | No activations |
| MocoTrack | `moco_track` (default) | Slowest | Highest |
| Static opt | `static_optimization` | Middle | Lower |

By default, Moco runs one motion at a time and `--num_workers` sets Ipopt threads (`0` = auto). Static optimization uses parallel motion workers like IK-only. Optional env `MOCO_NUM_THREADS` overrides auto-detection.

Each `.b3d` stores generalized coordinates plus custom channels: `guidance_features`, `sindy_features`, `muscle_activations` `[80, T]`, and (MocoTrack) `sim_grf` `[18, T]` plus `muscle_activation_mask` `[1, T]`.

At **20 fps**, segmented Moco uses **28-frame cores**, **3-frame buffers**, and **34-frame solve windows** (1.4 s / 0.14 s MinT timing).

**MocoTrack** (`moco_track`) — MinT-style segmented trajectory optimization with foot contact: IK → ground offset → 1.4 s Moco windows → seam stitch. Reference coordinates are low-pass filtered at **6 Hz inside Moco** (OpenCap). Failed segments leave **NaN gaps** in activations/GRF; the validity mask marks good frames. No pre-Moco q smoothing, IK frame interpolation, or post-Moco activation gap-filling.

**Static optimization** (`static_optimization`) — per-frame OpenSim static optimization on the same Rajagopal 80-muscle model with DeGroote muscles + reserves (no foot contact). Faster than Moco, lower fidelity.

OpenSim console output is **hidden by default** (`--opensim_log_level Off`).

Useful Moco flags: `--moco_core_duration_s`, `--moco_buffer_duration_s`, `--moco_stitch_blend_s`, `--moco_reference_lowpass_hz`, `--moco_states_speed_tracking_weight`, `--moco_no_reference_lowpass`, `--moco_mesh_interval`, `--moco_max_reserve_fraction`, `--moco_allow_high_reserve`, `--opensim_log_level`.

MinT smoke test (temp dir, self-deletes on success):

```bash
python scripts/smoke_moco_mint_preprocess.py
```

Unit tests (no OpenSim):

```bash
PYTHONPATH=. python3 -m unittest tests.test_moco_segment -v
```

**Kubernetes (64 worker pods + normalization pod):**

```bash
# Edit deploy/components/cluster-config/ for your image and PVC first — see deploy/README.md
./deploy/scripts/run-preprocess-nimble.sh static-optimization
./deploy/scripts/run-preprocess-nimble.sh none              # IK-only
./deploy/scripts/run-preprocess-nimble.sh moco-track
```

Local runs are unchanged (no sharding by default):

```bash
python scripts/preprocess_nimble.py --max_motions 1 \
  --opensim_log_level Warn
```

Distributed local test (optional):

```bash
python scripts/preprocess_nimble.py --max_motions 8 --num_shards 4 --shard_index 0 \
  --skip_normalization --activation_method none
# repeat shard_index 1..3, then:
python scripts/compute_normalization.py --num_shards 4
```

After upgrading the B3D schema (e.g. adding `muscle_activations`), **re-run preprocess** without `--skip_existing` on old caches.

### 2. Train SINDy

Requires B3D cache with **muscle activations** (preprocess with `moco_track` or `static_optimization`, not `none`).

```bash
python scripts/train_sindy.py --output results/sindy
```

Config: `configs/train_sindy.json`. Joint model predicts **103 channels** (23 bio + 80 muscles) from text-conditioned sparse `Ξ(text)`. Ground-truth targets come from cached `guidance_features` and `muscle_activations`. Old `target_dim=23` checkpoints are incompatible.

### 3. Train activation surrogate

```bash
python scripts/train_surrogate.py --config configs/train_surrogate.json --output results/activation_surrogate
```

Config: `configs/train_surrogate.json`. Default architecture is a **temporal transformer** (fidelity-first). Training uses L1 on all frames in each window (plus optional temporal regularization via `lambda_temporal`). Motions whose cached `muscle_activations` are all-zero placeholders are skipped by default (`skip_zero_placeholders=1`).

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

**Container image:** GitHub Actions publishes `ghcr.io/rose-stl-lab/sindyffuse:latest` from `env/Dockerfile` (GHCR container storage is currently free; see deploy docs for size notes).

## Project layout

| Path | Role |
|------|------|
| `scripts/preprocess_nimble.py` | HumanML3D → Nimble B3D cache |
| `scripts/compute_normalization.py` | Merge shard manifests; compute `Mean.npy` / `Std.npy` |
| `scripts/train_sindy.py` | Train SINDy text→Xi model |
| `scripts/train_surrogate.py` | Train q→activation surrogate |
| `scripts/train_diffusion.py` | Train text-conditioned diffusion |
| `scripts/generate_motion.py` | Sample motion from trained diffusion |
| `env/environment.yaml` | Conda environment |
| `env/Dockerfile` | Container image (`docker build -f env/Dockerfile .`; CI publishes to GHCR) |
| `deploy/` | Kubernetes job manifests (see `deploy/README.md`) |
| `nimble/` | IK, B3D I/O, OpenSim muscle activation, Rajagopal guidance |
| `surrogate/` | Differentiable activation surrogate (ML) |
| `sindy/` | SINDy library, dataset, training |
| `diffusion/` | Text-conditioned motion diffusion |
| `datasets/` | HumanML3D loaders (Python only; data is local) |
| `tests/` | Smoke tests |

## Tests

```bash
conda activate sindyffuse
cd /path/to/SINDyffuse
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

OpenSim-backed tests (`test_muscle_activation`) require the `sindyffuse` conda env. Full-body Moco is **opt-in** (slow):

```bash
RUN_MOCO_SMOKE=1 PYTHONPATH=. python3 -m unittest tests.test_muscle_activation -v
```

## Troubleshooting

```bash
python scripts/preprocess_nimble.py --max_motions 1 --opensim_log_level Warn
```

- Re-run `scripts/preprocess_nimble.py` after upgrading B3D schema (e.g. adding `muscle_activations`).  
- If Ctrl+C does not stop Moco: `pkill -9 -f "python scripts/preprocess_nimble.py"`.
