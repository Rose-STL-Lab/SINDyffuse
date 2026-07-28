# SINDyffuse

Text-conditioned human motion diffusion with **SINDy** biomechanics targets and musculoskeletal physics guidance.

HumanML3D joint trajectories are retargeted to the **Lai lower-body + Bruno thoracolumbar** OpenSim models, cached as NPZ files, and used to train:

1. **SINDy** — text → sparse coefficients for **424 targets** (22 L_bio + 402 muscle activations)
2. **Activation surrogate** — fast MinT `q` → 402 muscle activations (MinT labels via `musint`)
3. **Diffusion** — text → motion with SINDy guidance (`loss_diff + lambda_sindy * loss_sindy`)

## Setup

```bash
conda env create -f env/environment.yaml
conda activate sindyffuse
```

OpenSim comes from the `opensim` conda package. The MinT Lai model is bundled under `osim/models/`; set `MINT_ROOT` to your extracted MinT dataset for precomputed muscle labels.

```bash
export HUMANML3D_ROOT=/path/to/HumanML3D   # optional; default: datasets/HumanML3D
export MINT_ROOT=/path/to/MinT             # required for MinT muscle labels at preprocess
export SMPLH_MODEL_PATH=/path/to/smplh     # SMPLH_NEUTRAL.pkl for MinT-faithful retargeting
```

### Retargeting (HumanML3D → MinT `q`)

Default preprocess uses the **MinT-faithful** pipeline:

```
HML 22×3 joints → SMPL-H fit (smplx) → 67 SSM virtual markers → OpenSim IK (Lai model)
```

Requires `SMPLH_MODEL_PATH` and OpenSim from conda. Marker sets and IK setup are generated under `osim/models/opensim/` on first run.

| Env / flag | Purpose |
|------------|---------|
| `SMPLH_MODEL_PATH` | SMPL-H model directory (required for `method=mint`) |
| `MINT_RETARGET_METHOD` | `mint` (default) or `bootstrap` |
| `MINT_IK_KEEP_WORKDIR=1` | Keep temp IK work dirs for debugging |

## Data layout

```
datasets/HumanML3D/          # not committed (~20GB)
  new_joint_vecs/
  texts/
  train.txt, val.txt, test.txt
  mint_cache/                # MinT NPZ cache
    {motion_id}.npz          # q, muscle_activations [402], guidance/sindy features
    Mean.npy, Std.npy
    metadata.json
```

## Pipeline

Run entry points from the repo root:

```bash
cd /path/to/SINDyffuse
export MINT_ROOT=/path/to/MinT
```

### 1. Preprocess (MinT retarget + musint labels)

```bash
python scripts/preprocess_mint.py --max_motions 1
python scripts/compute_normalization.py
```

MinT activations are loaded from `musint` at **20 fps**. Motions without MinT overlap get `has_mint_labels=false` and are skipped by surrogate/SINDy training by default.

**Kubernetes:**

```bash
./deploy/scripts/run-preprocess-dataset.sh
./deploy/scripts/run-normalize-dataset.sh
```

### 2. Train SINDy (424 targets)

```bash
python scripts/train_sindy.py --config configs/train_sindy.json
```

### 3. Train activation surrogate (q → 402)

```bash
python scripts/train_surrogate.py --config configs/train_surrogate.json
```

### 4. Train diffusion (MinT ndof)

```bash
python scripts/train_diffusion.py --config configs/train_diffusion.json
```

### 5. Generate motion

```bash
python scripts/generate_motion.py --checkpoint results/diffusion/latest.pt \
  --caption "a person walks forward" --out_npz out.npz \
  --guidance sindy \
  --sindy_checkpoint_dir results/sindy/latest \
  --surrogate_checkpoint_dir results/activation_surrogate/latest
```

## Kubernetes

```bash
kubectl apply -k deploy/jobs/preprocess-dataset
kubectl apply -k deploy/jobs/normalize-dataset
kubectl apply -k deploy/jobs/train-surrogate
kubectl apply -k deploy/jobs/train-sindy
kubectl apply -k deploy/jobs/train-diffusion/sindy

kubectl apply -k deploy/dev
kubectl exec -it sindyffuse-dev -- bash -l
```

See [deploy/README.md](deploy/README.md) for image build, storage setup, and job details.

## Project layout

| Path | Role |
|------|------|
| `common/` | Shared paths, logging, distributed training helpers |
| `configs/` | Training JSON configs |
| `datasets/` | HumanML3D loaders (Python only; data is local) |
| `deploy/` | Kubernetes job manifests |
| `diffusion/` | Text-conditioned motion diffusion |
| `env/` | Conda environment and Dockerfile |
| `eval/` | Smoke tests and temporary eval outputs (`eval/tmp/`) |
| `osim/` | OpenSim integration, retargeting, bundled models |
| `scripts/` | CLI entry points |
| `sindy/` | SINDy library, dataset, training |
| `surrogate/` | Differentiable activation surrogate |

## Tests

```bash
conda activate sindyffuse
cd /path/to/SINDyffuse
PYTHONPATH=. python3 -m unittest discover -s eval/tests -v
```
