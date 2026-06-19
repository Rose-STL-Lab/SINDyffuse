# Kubernetes deployment

SINDyffuse runs locally via `scripts/*.py` and on Kubernetes via Job manifests in `deploy/jobs/`. Each job wraps the same Python entry points with cluster resources (CPU, GPU, PVC).

## Layout

```
deploy/
  pvc.yaml                    # cluster PVC (apply once)
  components/
    cluster-config/           # edit image + PVC here (applies to all jobs/dev pod)
  jobs/
    preprocess-static-optimization/
    benchmark-moco-parallel/
    train-sindy/
    train-diffusion-none/
    train-diffusion-sindy/
    train-diffusion-nimble/
  dev-pod/                  # long-running interactive pod
```

| Local | Kubernetes |
|-------|------------|
| `python scripts/preprocess_nimble.py ...` | `kubectl apply -k deploy/jobs/preprocess-static-optimization` |
| `python scripts/benchmark_moco_parallel.py ...` | `kubectl apply -k deploy/jobs/benchmark-moco-parallel` |
| `python scripts/train_sindy.py ...` | `kubectl apply -k deploy/jobs/train-sindy` |
| `python scripts/train_diffusion.py ...` | `kubectl apply -k deploy/jobs/train-diffusion-{none,sindy,nimble}` |

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

After the first publish, open the package on GitHub (**Packages → sindyffuse → Package settings**) and set visibility to **Public** so clusters can pull without credentials.

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
| `kustomization.yaml` | `images.newName`, `images.newTag` |
| `pvc-patch.yaml`, `pvc-patch-pod.yaml` | `claimName` (PVC name) |

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

- Repo checkout at `/workspace/SINDyffuse`
- HumanML3D at `/workspace/SINDyffuse/datasets/HumanML3D`

### 4. Apply a job

```bash
# Preprocess (static optimization, single pod, parallel workers)
kubectl apply -k deploy/jobs/preprocess-static-optimization
kubectl get pods -l job-name=sindyffuse-preprocess-nimble -w

# Train SINDy (2× GPU)
kubectl apply -k deploy/jobs/train-sindy

# Train diffusion (pick guidance mode)
kubectl apply -k deploy/jobs/train-diffusion-nimble
```

Preview rendered manifests:

```bash
kubectl kustomize deploy/jobs/preprocess-static-optimization
```

Delete a job before re-running:

```bash
kubectl delete job sindyffuse-preprocess-nimble --ignore-not-found
```

## Dev pod

Long-running pod for interactive work on the cluster (same image + PVC as jobs):

```bash
kubectl apply -k deploy/dev-pod
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

Default resources: 8–32 CPU, 32–64Gi memory. Edit `deploy/dev-pod/pod.yaml` to add GPUs or change limits.

## In-container layout

All manifests assume the PVC is mounted at `/workspace` with the repo at `/workspace/SINDyffuse` (`workingDir` on every pod/job). HumanML3D lives at `/workspace/SINDyffuse/datasets/HumanML3D`.

Only **preprocess** exposes optional env overrides: `MAX_MOTIONS`, `SKIP_EXISTING`, `PREPROCESS_NUM_WORKERS`.

## Resource profiles

| Job | CPUs | Memory | GPUs |
|-----|------|--------|------|
| dev-pod | 8–32 | 32–64Gi | — (add in pod.yaml if needed) |
| preprocess-static-optimization | 64 | 64Gi | — |
| benchmark-moco-parallel | 64 | 64Gi | — |
| train-sindy | 16 | 64Gi | 2 |
| train-diffusion-* | 16 | 64Gi | 2 |

Edit `deploy/jobs/<name>/job.yaml` to match your nodes.

## Pipeline order

1. `preprocess-static-optimization` (or run `scripts/preprocess_nimble.py` locally)
2. `train-sindy`
3. `train-surrogate` — local only for now (`scripts/train_surrogate.py`)
4. `train-diffusion-{none,sindy,nimble}`

## Local development

No Kubernetes required for development:

```bash
conda env create -f env/environment.yaml
conda activate sindyffuse
python scripts/preprocess_nimble.py --max_motions 1
```

See the root [README.md](../README.md) for the full local pipeline.
