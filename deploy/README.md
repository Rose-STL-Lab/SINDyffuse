# Kubernetes deployment

SINDyffuse runs locally via `scripts/*.py` and on Kubernetes via Job manifests in `deploy/jobs/`. Each job wraps the same Python entry points with cluster resources (CPU, GPU, PVC).

## Layout

```
deploy/
  pvc.yaml                    # cluster PVC (apply once)
  scripts/
    run-preprocess-nimble.sh  # preprocess worker + normalization pipeline
  components/
    cluster-config/           # edit image + PVC here (applies to all jobs/dev pod)
  jobs/
    preprocess-mint/            # MinT NPZ cache (default on mint branch)
    train-surrogate-mint/
    train-sindy-mint/
    train-diffusion-mint/
      sindy/
    preprocess-nimble/
      none/                   # IK-only (--activation_method none)
      static-optimization/
      moco-track/
      normalization/          # Mean.npy / Std.npy after workers
    benchmark-moco-parallel/
    train-sindy/
    train-surrogate/
    train-diffusion/
      none/
      nimble/
      sindy/
  dev/                        # long-running interactive pod
```

| Local | Kubernetes |
|-------|------------|
| `python scripts/preprocess_mint.py ...` | `./deploy/scripts/run-preprocess-mint.sh` or `kubectl apply -k deploy/jobs/preprocess-mint` |
| `python scripts/preprocess_nimble.py ...` | `./deploy/scripts/run-preprocess-nimble.sh [none\|static-optimization\|moco-track]` |
| `python scripts/compute_normalization.py ...` | `kubectl apply -k deploy/jobs/preprocess-nimble/normalization` |
| `python scripts/benchmark_moco_parallel.py ...` | `kubectl apply -k deploy/jobs/benchmark-moco-parallel` |
| `python scripts/train_surrogate.py --config configs/train_surrogate_mint.json` | `kubectl apply -k deploy/jobs/train-surrogate-mint` |
| `python scripts/train_sindy.py --config configs/train_sindy_mint.json` | `kubectl apply -k deploy/jobs/train-sindy-mint` |
| `python scripts/train_diffusion.py --config configs/train_diffusion_mint.json` | `kubectl apply -k deploy/jobs/train-diffusion-mint/sindy` |
| `python scripts/train_sindy.py ...` | `kubectl apply -k deploy/jobs/train-sindy` |
| `python scripts/train_surrogate.py ...` | `kubectl apply -k deploy/jobs/train-surrogate` |
| `python scripts/train_diffusion.py ...` | `kubectl apply -k deploy/jobs/train-diffusion/{none,nimble,sindy}` |

## Container image (GHCR)

The SINDyffuse runtime image is built from `env/Dockerfile` (CUDA + conda env `sindyffuse` from `env/environment.yaml`).

### Published image

GitHub Actions publishes to **GitHub Container Registry** when `env/Dockerfile` or `env/environment.yaml` changes on `main`, or when you push a version tag (`v*`):

```
ghcr.io/rose-stl-lab/sindyffuse:latest
ghcr.io/rose-stl-lab/sindyffuse:sha-<commit>
ghcr.io/rose-stl-lab/sindyffuse:v1.0.0   # on git tag v1.0.0
```

Pull (after the package is public, or with `read:packages` auth):

```bash
docker pull ghcr.io/rose-stl-lab/sindyffuse:latest
```

Trigger a manual build: **Actions → Docker image → Run workflow**.

After the first publish, open the package on GitHub (**Packages → sindyffuse → Package settings**) and set visibility to **Public** so clusters can pull without credentials. If your organization disables public packages, use a pull secret instead (see [Quick start §2](#2-configure-cluster-settings)).

### GitHub limits (free accounts)

| Limit | Applies to SINDyffuse image? |
|-------|------------------------------|
| **500 MB** GitHub Packages storage (npm, etc.) | **No** — GHCR containers are billed separately |
| **GHCR storage/bandwidth** | **Currently free** (GitHub may change with 1-month notice) |
| **10 GB max per image layer** | Yes — conda + PyTorch layers must stay under this (typical builds do) |
| **10 minute upload timeout per layer** | Yes — first push can be slow; rebuilds use GHA cache |

Expect a **large image** (~8–15 GB total, CUDA + PyTorch + OpenSim + Nimble). That is normal for this stack and is fine on GHCR today.

**CI build failed with `No space left on device`?** GitHub's default runners have limited disk; the workflow runs a free-disk-space step before building. Re-run **Actions → Docker image** after pulling the latest workflow fix.

### Build locally

From the repo root:

```bash
docker build -f env/Dockerfile -t ghcr.io/rose-stl-lab/sindyffuse:latest .
docker push ghcr.io/rose-stl-lab/sindyffuse:latest   # requires GHCR login
```

## Quick start

### 1. Use the published image (or build locally)

Default image (from GHCR, after CI has run once):

```
ghcr.io/rose-stl-lab/sindyffuse:latest
```

Or build yourself:

```bash
docker build -f env/Dockerfile -t ghcr.io/rose-stl-lab/sindyffuse:latest .
docker push ghcr.io/rose-stl-lab/sindyffuse:latest   # optional
```

See [Container image (GHCR)](README.md#container-image-ghcr) for limits and visibility settings.

### 2. Configure cluster settings

Edit **`deploy/components/cluster-config/`** (single template for every manifest):

| File | Set |
|------|-----|
| `kustomization.yaml` | `images.newName`, `images.newTag` (default: `ghcr.io/rose-stl-lab/sindyffuse:latest`) |
| `pvc-patch.yaml`, `pvc-patch-pod.yaml` | `claimName` (PVC name) |

All jobs and the dev pod pull **`ghcr.io/rose-stl-lab/sindyffuse:latest`** from CI. There is no Docker Hub image in this repo's deploy path.

**GHCR auth on Nautilus (required when the org package is private):** create a pull secret once per namespace:

```bash
# GitHub → Settings → Developer settings → Personal access tokens → read:packages
kubectl create secret docker-registry ghcr-secret -n ai-md \
  --docker-server=ghcr.io \
  --docker-username=YOUR_GITHUB_USER \
  --docker-password=YOUR_GITHUB_PAT
```

`cluster-config` already references this secret (`ghcr-pull-secret-patch.yaml`). After creating the secret, delete old jobs and re-apply so nodes pull the CI-built image:

```bash
kubectl delete job sindyffuse-preprocess-mint -n ai-md --ignore-not-found
kubectl apply -k deploy/jobs/preprocess-mint/workers -n ai-md
```

If your organization later allows **Public** visibility on the package, remove the `ghcr-pull-secret-patch.yaml` patches from `cluster-config/kustomization.yaml` so anonymous pulls work without a secret.

To keep local overrides out of git, copy the folder and point a job’s `kustomization.yaml` at your copy:

```bash
cp -r deploy/components/cluster-config deploy/overlays/local/cluster-config
# components: ../overlays/local/cluster-config
```

### 3. Create storage (if needed)

Apply once per cluster:

```bash
kubectl apply -f deploy/pvc.yaml
```

The PVC should contain:

- Repo checkout at `/mnt/SINDyffuse`
- HumanML3D at `/mnt/SINDyffuse/datasets/HumanML3D`

### 4. Apply a job

```bash
# Preprocess — pick activation method (default: static-optimization)
./deploy/scripts/run-preprocess-nimble.sh none
./deploy/scripts/run-preprocess-nimble.sh static-optimization
./deploy/scripts/run-preprocess-nimble.sh moco-track

# Or run normalization alone after workers already finished:
kubectl apply -k deploy/jobs/preprocess-nimble/normalization

kubectl get pods -l activation-method=static-optimization -w

# Train SINDy (2× GPU)
kubectl apply -k deploy/jobs/train-sindy

# Train activation surrogate (2× GPU; auto torchrun in Python; requires Mean.npy/Std.npy)
kubectl apply -k deploy/jobs/train-surrogate

# Train diffusion — pick guidance mode
kubectl apply -k deploy/jobs/train-diffusion/none
kubectl apply -k deploy/jobs/train-diffusion/nimble
kubectl apply -k deploy/jobs/train-diffusion/sindy
```

Preview a rendered preprocess worker manifest:

```bash
kubectl kustomize deploy/jobs/preprocess-nimble/static-optimization
```

Delete jobs before a manual re-run (example for static-optimization):

```bash
kubectl delete job sindyffuse-preprocess-static-optimization \
  sindyffuse-compute-normalization --ignore-not-found
```

## Dev pod

Long-running pod for interactive work on the cluster (same image + PVC as jobs):

```bash
kubectl apply -k deploy/dev
kubectl get pod sindyffuse-dev -w
kubectl exec -it sindyffuse-dev -- bash -l
```

Inside the shell, `conda activate sindyffuse` is already configured. Run scripts the same way as locally:

```bash
python scripts/preprocess_nimble.py --max_motions 1
```

Delete when done:

```bash
kubectl delete pod sindyffuse-dev
```

Default resources: 8–32 CPU, 32–64Gi memory. Edit `deploy/dev/pod.yaml` to add GPUs or change limits.

## In-container layout

All manifests assume the PVC is mounted at `/mnt` with the repo at `/mnt/SINDyffuse` (`workingDir` on every pod/job). HumanML3D lives at `/mnt/SINDyffuse/datasets/HumanML3D`.

Only **preprocess** is orchestrated via [`deploy/scripts/run-preprocess-nimble.sh`](scripts/run-preprocess-nimble.sh). Optional env: `PREPROCESS_NUM_SHARDS`, `MAX_MOTIONS`, `SKIP_EXISTING` (set in the job yaml).

Each preprocess variant uses an Indexed worker Job (`parallelism=completions=64`). Normalization is a separate Job under `preprocess-nimble/normalization` (`compute_normalization.py`) that merges shard manifests and writes `nimble_b3d/Mean.npy` / `Std.npy`. `run-preprocess-nimble.sh` runs workers then normalization; you can also apply normalization alone.

## Resource profiles

| Job | CPUs | Memory | GPUs |
|-----|------|--------|------|
| dev | 8–32 | 32–64Gi | — (add in pod.yaml if needed) |
| preprocess-nimble/* (workers) | 64 × 1 | 64 × 2–4Gi | — (CPU-only OpenSim/Nimble) |
| preprocess-nimble/normalization | 1 | 2Gi | — |
| benchmark-moco-parallel | 64 | 64Gi | — |
| train-sindy | 16 | 64Gi | 1 |
| train-surrogate | 16 | 64Gi | 1 |
| train-diffusion/* | 16 | 64Gi | 1 |

Edit `deploy/jobs/<name>/job.yaml` to match your nodes.

### Multi-GPU training

Training jobs source `deploy/scripts/job-env.sh` (conda + runtime env) and invoke the same Python entry points used locally, e.g. `python scripts/train_diffusion.py --guidance sindy --preload`. Run directories, resolved configs, preflight checks, and `latest` symlinks are handled inside the scripts (`common/run_setup.py`).

Each trainer calls `maybe_relaunch_with_torchrun()` in `common/distributed.py`, which:

1. Reads `NPROC_PER_NODE` (set in job yaml to match `nvidia.com/gpu`, default **1**)
2. Else counts `CUDA_VISIBLE_DEVICES`
3. Else uses `torch.cuda.device_count()`
4. Caps the process count by the number of CUDA devices that can actually run kernels
5. Re-execs under `torchrun` when count > 1 and `distributed.enabled` is not `false`

If the pod only exposes one working GPU, training automatically falls back to single-GPU mode instead of launching a broken 2-process job.

Set `NPROC_PER_NODE=1` or `SINDYFFUSE_NO_TORCHRUN=1` to force single-process training.
Startup logs include `[distributed/gpu]` with rank, world size, device count, and `/dev/nvidia*` nodes.

## Pipeline order

1. `preprocess-nimble/{none,static-optimization,moco-track}` workers via `run-preprocess-nimble.sh` (or `scripts/preprocess_nimble.py` locally)
2. `preprocess-nimble/normalization` (included in the script, or `kubectl apply -k deploy/jobs/preprocess-nimble/normalization`)
3. `train-sindy`
4. `train-surrogate`
5. `train-diffusion/{none,nimble,sindy}`

## Results layout

Each trainer writes timestamped runs under ``results/<family>/runs/`` and updates ``results/<family>/latest`` when training completes (see ``common/run_setup.py``).

## Local development

No Kubernetes required for development:

```bash
conda env create -f env/environment.yaml
conda activate sindyffuse
python scripts/preprocess_nimble.py --max_motions 1
```

See the root [README.md](../README.md) for the full local pipeline.
