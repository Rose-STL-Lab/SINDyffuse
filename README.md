# SINDyffuse

Text-conditioned human motion diffusion with **SINDy** biomechanics targets and **Nimble/OpenSim** physics guidance.

HumanML3D joint trajectories are retargeted to the Rajagopal 2015 musculoskeletal model, cached as Nimble B3D files, and used to train:

1. **SINDy** — text → sparse biomechanics coefficients  
2. **Activation surrogate** — fast `q` → 80 muscle activations (OpenSim labels at preprocess)  
3. **Diffusion** — text → motion with optional Nimble/SINDy guidance  

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

Each `.b3d` stores generalized coordinates plus custom channels: `guidance_features`, `sindy_features`, `muscle_activations` `[80, T]`, and `muscle_activation_mask` `[1, T]` (1.0 = valid label at that frame; 0.0 = failed frame).

**MocoTrack** (`moco_track`) — one-pass trajectory optimization with foot contact (20 fps tuned): 0.05 s mesh (1 node/frame), **6 Hz reference + IK low-pass** (OpenCap), **uniform tracking weight 1** on all coordinates except `pelvis_ty` (contact-driven), adaptive refine to 0.02 s, tol 0.01, implicit-muscle auxiliary minimization.

**Static optimization** (`static_optimization`) — per-frame OpenSim static optimization on the same Rajagopal 80-muscle model with DeGroote muscles + reserves (no foot contact). Faster than Moco, lower fidelity.

OpenSim console output is **hidden by default** (`--opensim_log_level Off`).

Useful Moco flags: `--moco_reference_lowpass_hz`, `--moco_states_speed_tracking_weight`, `--moco_aux_coord_tracking_weight`, `--moco_no_reference_lowpass`, `--moco_no_apply_tracked_guess`, `--moco_mesh_interval`, `--moco_states_tracking_weight`, `--moco_max_reserve_fraction`, `--moco_allow_high_reserve`, `--moco_reserve_scale`, `--moco_min_success_fraction` (default 0.5), `--moco_allow_low_valid`, `--moco_no_repair`, `--moco_no_adaptive_mesh`, `--moco_min_frames`, `--moco_min_ik_success_fraction`, `--moco_max_pelvis_ty_range_m`, `--opensim_log_level`.

**Kubernetes pilot (5 motions):**

```bash
kubectl apply -f env/preprocess_nimble.yaml
kubectl logs -f job/sindyffuse-preprocess-nimble
```

Local equivalent:

```bash
python scripts/preprocess_nimble.py --max_motions 1 \
  --opensim_log_level Warn --moco_allow_low_valid
```

After upgrading the B3D schema (e.g. adding `muscle_activation_mask`), **re-run preprocess** without `--skip_existing` on old caches.

### 2. Train SINDy

```bash
python scripts/train_sindy.py --output results/sindy
```

Config: `configs/train_sindy.json`

### 3. Train activation surrogate

```bash
python scripts/train_surrogate.py --config configs/train_surrogate.json --output results/activation_surrogate
```

Config: `configs/train_surrogate.json`. Training uses **masked L1** on frames where `muscle_activation_mask == 1` (see `min_window_valid_fraction`, `require_activation_mask` in the config). Re-preprocess with the current pipeline before setting `require_activation_mask: 1`.

### 4. Train diffusion

```bash
python scripts/train_diffusion.py --config configs/train_diffusion.json --out_dir results/diffusion
```

Config: `configs/train_diffusion.json`

### 5. Generate motion

```bash
python scripts/generate_motion.py --prompt "a person walks forward"
```

## Project layout

| Path | Role |
|------|------|
| `scripts/preprocess_nimble.py` | HumanML3D → Nimble B3D cache |
| `scripts/train_sindy.py` | Train SINDy text→Xi model |
| `scripts/train_surrogate.py` | Train q→activation surrogate |
| `scripts/train_diffusion.py` | Train text-conditioned diffusion |
| `scripts/generate_motion.py` | Sample motion from trained diffusion |
| `env/environment.yaml` | Conda environment |
| `env/Dockerfile` | Container image (`docker build -f env/Dockerfile .`) |
| `env/*.yaml` | Kubernetes job manifests |
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
python scripts/preprocess_nimble.py --max_motions 1 --opensim_log_level Warn --moco_allow_low_valid
```

- Re-run `scripts/preprocess_nimble.py` after upgrading B3D schema (e.g. adding `muscle_activations` or `muscle_activation_mask`).  
- If Ctrl+C does not stop Moco: `pkill -9 -f "python scripts/preprocess_nimble.py"`.
