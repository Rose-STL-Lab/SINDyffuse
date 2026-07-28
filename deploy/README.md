# Kubernetes deployment

SINDyffuse runs locally via `scripts/*.py` and on Kubernetes via Job manifests in `deploy/jobs/`. Each job wraps the same Python entry points with cluster resources (CPU, GPU, PVC).

## Layout

```
deploy/
  pvc.yaml                    # cluster PVC (apply once)
  scripts/
    run-preprocess-dataset.sh # indexed preprocess workers
    run-normalize-dataset.sh  # Mean.npy / Std.npy after workers
    job-env.sh
  components/
    cluster-config/           # edit image + PVC here (applies to all jobs/dev pod)
  jobs/
    preprocess-dataset/
    normalize-dataset/
    train-surrogate/
    train-sindy/
    train-diffusion/
  dev/                        # long-running interactive pod
```

| Local | Kubernetes |
|-------|------------|
| `python scripts/preprocess_mint.py ...` | `./deploy/scripts/run-preprocess-dataset.sh` |
| `python scripts/compute_normalization.py ...` | `./deploy/scripts/run-normalize-dataset.sh` |
| `python scripts/train_surrogate.py --config configs/train_surrogate.json` | `kubectl apply -k deploy/jobs/train-surrogate` |
| `python scripts/train_sindy.py --config configs/train_sindy.json` | `kubectl apply -k deploy/jobs/train-sindy` |
| `python scripts/train_diffusion.py --config configs/train_diffusion.json` | `kubectl apply -k deploy/jobs/train-diffusion` |

## Container image (Docker Hub)

The SINDyffuse runtime image is built from `env/Dockerfile` (CUDA + conda env `sindyffuse` from `env/environment.yaml`).

Published image (after CI runs on `main`):

```
ncking/sindyffuse:latest
```

Expect a **large image** (~8–15 GB total, CUDA + PyTorch + OpenSim).

Build locally:

```bash
docker build -f env/Dockerfile -t ncking/sindyffuse:latest .
```

## Quick start

### 1. Configure cluster settings

Edit **`deploy/components/cluster-config/`**:

| File | Set |
|------|-----|
| `kustomization.yaml` | `images.newName`, `images.newTag` |
| `pvc-patch.yaml`, `pvc-patch-pod.yaml` | `claimName` |

### 2. Create storage (if needed)

```bash
kubectl apply -f deploy/pvc.yaml
```

The PVC should contain the repo at `/mnt/SINDyffuse` and HumanML3D at `/mnt/SINDyffuse/datasets/HumanML3D`.

### 3. Run the pipeline

```bash
./deploy/scripts/run-preprocess-dataset.sh
./deploy/scripts/run-normalize-dataset.sh

kubectl apply -k deploy/jobs/train-sindy
kubectl apply -k deploy/jobs/train-surrogate
kubectl apply -k deploy/jobs/train-diffusion
```

## Dev pod

```bash
kubectl apply -k deploy/dev
kubectl exec -it sindyffuse-dev -- bash -l
python scripts/preprocess_mint.py --max_motions 1
```

## Pipeline order

1. `preprocess-dataset` workers (via `run-preprocess-dataset.sh` or `scripts/preprocess_mint.py`)
2. `normalize-dataset` (`run-normalize-dataset.sh` or `scripts/compute_normalization.py`)
3. `train-sindy`
4. `train-surrogate`
5. `train-diffusion`

See the root [README.md](../README.md) for the full local pipeline.
