# Kubernetes deployment

SINDyffuse runs locally via `scripts/*.py` and on Kubernetes via Job manifests in `deploy/jobs/`. Each job wraps the same Python entry points with cluster resources (CPU, GPU, PVC).

## Layout

```
deploy/
  pvc.yaml                    # cluster PVC (apply once)
  scripts/
    run-preprocess-dataset.sh # full pipeline or individual stages
    run-path-fit.sh           # optional laptop driver for path-fit pipeline
    k8s-orchestrate-lib.sh    # shared kubectl wait helpers
    path-fit-orchestrate.sh   # in-cluster path-fit orchestrator
    moco-track-orchestrate.sh # moco workers → normalization
    preprocess-dataset-orchestrate.sh  # IK → path-fit → moco orchestrators
  components/
    cluster-config/           # edit image + PVC here (applies to all jobs/dev pod)
  jobs/
    preprocess-dataset/
      components/orchestrator-rbac/  # shared ServiceAccount + Role
      orchestrator/           # full pipeline: IK → path-fit → moco orchestrators
      ik/                       # IndexedJob (180 × 1 CPU)
      fit-function-paths/
        orchestrator/           # prepare → convert → fit
        prepare/
        convert/                # IndexedJob B3D → .mot workers
        fit/                    # merge + PolynomialPathFitter
      moco-track/
        job.yaml                # IndexedJob MocoTrack workers
        orchestrator/           # moco workers → normalization
      normalization/            # Mean.npy / Std.npy after moco workers
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
| `python scripts/preprocess_ik.py ...` then `preprocess_moco.py` | `./deploy/scripts/run-preprocess-dataset.sh [full\|ik\|path-fit\|moco]` |
| `python scripts/compute_normalization.py ...` | via moco-track orchestrator, or `kubectl apply -k deploy/jobs/preprocess-dataset/normalization` |
| `python scripts/fit_rajagopal_function_paths.py ...` (Mode A local) | `kubectl apply -k deploy/jobs/preprocess-dataset/fit-function-paths/orchestrator` (Mode C) or `./deploy/scripts/run-path-fit.sh` |
| `python scripts/benchmark_moco_parallel.py ...` | `kubectl apply -k deploy/jobs/benchmark-moco-parallel` |
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

## Nautilus (NRP) compliance

Apply jobs to your namespace (not stored in repo manifests):

```bash
kubectl apply -k deploy/jobs/preprocess-dataset/ik -n YOUR_NAMESPACE
```

Checklist aligned with [NRP cluster policies](https://nrp.ai/documentation/userdocs/start/policies/):

| Rule | Our setup |
|------|-----------|
| **limits within 20% of requests** | All jobs use **limits = requests** (Guaranteed QoS) |
| **> ~100 pods: limit = request** | Moco (180 shards): 10 CPU / 16Gi limits = requests ✓ |
| **1 CPU + 2Gi exemption** | IK workers (180 × 1 CPU, 2Gi): exempt from utilization violation checks ✓ |
| **Batch jobs, not sleep infinity** | Jobs run Python scripts to completion ✓ |
| **No GPUs on CPU-only preprocess** | Preprocess jobs request CPU/memory only ✓ |
| **PVC** | `rook-cephfs`, `ReadWriteMany` (standard Nautilus CephFS) ✓ |
| **Large parallel submits** | 180 moco pods is a large footprint; coordinate with namespace admins if scheduling is slow |

**backoffLimit:** Indexed preprocess jobs retry failed shard pods a limited number of times before the Job fails (IK: 32; moco: 64; path-fit convert: 32). Single-pod jobs (normalization, path-fit prepare/fit/orchestrator): 3 (orchestrator: 1).

**SKIP_EXISTING:** Not set in job manifests (default: reprocess all motions). To skip existing B3D files on retry, set env `SKIP_EXISTING=1` at apply time or add it to your local overlay.

**Image:** `deploy/components/cluster-config` rewrites `sindyffuse:latest` → `ncking/sindyffuse:latest`. Every job kustomization must include that component.

Edit **`deploy/components/cluster-config/`** for registry tag and PVC name for your environment.

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

- Repo checkout at `/mnt/SINDyffuse`
- HumanML3D at `/mnt/SINDyffuse/datasets/HumanML3D`

### 4. Apply a job

```bash
# Full preprocess pipeline (IK → path fit → MocoTrack → normalization)
./deploy/scripts/run-preprocess-dataset.sh full YOUR_NAMESPACE

# Or individual stages:
./deploy/scripts/run-preprocess-dataset.sh ik YOUR_NAMESPACE
./deploy/scripts/run-preprocess-dataset.sh path-fit YOUR_NAMESPACE
./deploy/scripts/run-preprocess-dataset.sh moco YOUR_NAMESPACE

# Or apply orchestrators directly:
kubectl apply -k deploy/jobs/preprocess-dataset/orchestrator -n YOUR_NAMESPACE
kubectl apply -k deploy/jobs/preprocess-dataset/fit-function-paths/orchestrator -n YOUR_NAMESPACE
kubectl apply -k deploy/jobs/preprocess-dataset/moco-track/orchestrator -n YOUR_NAMESPACE

# Train SINDy (2× GPU)
kubectl apply -k deploy/jobs/train-sindy

# Train activation surrogate (2× GPU; auto torchrun in Python; requires Mean.npy/Std.npy)
kubectl apply -k deploy/jobs/train-surrogate

# Train diffusion — pick guidance mode
kubectl apply -k deploy/jobs/train-diffusion/none
kubectl apply -k deploy/jobs/train-diffusion/nimble
kubectl apply -k deploy/jobs/train-diffusion/sindy
```

Preview a rendered preprocess manifest:

```bash
kubectl kustomize deploy/jobs/preprocess-dataset/ik
```

Delete jobs before a manual re-run (example for full pipeline orchestrator):

```bash
kubectl delete job sindyffuse-preprocess-dataset-orchestrator \
  sindyffuse-preprocess-ik \
  sindyffuse-path-fit-orchestrator \
  sindyffuse-moco-track-orchestrator \
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
python scripts/preprocess_ik.py --max_motions 1
python scripts/preprocess_moco.py --max_motions 1
```

Delete when done:

```bash
kubectl delete pod sindyffuse-dev
```

Default resources: 8–32 CPU, 32–64Gi memory. Edit `deploy/dev/pod.yaml` to add GPUs or change limits.

## In-container layout

All manifests assume the PVC is mounted at `/mnt` with the repo at `/mnt/SINDyffuse` (`workingDir` on every pod/job). HumanML3D lives at `/mnt/SINDyffuse/datasets/HumanML3D`.

Only **preprocess** jobs accept optional env: `PREPROCESS_NUM_SHARDS`, `PATH_FIT_NUM_SHARDS`, `MAX_MOTIONS`, `SKIP_EXISTING` (omit `SKIP_EXISTING` to reprocess all motions).

Preprocess uses Indexed worker Jobs (`parallelism=completions=180` for IK, path-fit convert, and moco). Normalization is a separate Job under `preprocess-dataset/normalization` (`compute_normalization.py`) that merges moco shard manifests and writes `nimble_b3d/Mean.npy` / `Std.npy`. The moco-track orchestrator runs workers then normalization; you can also apply normalization alone after moco workers finish.

## Resource profiles

| Job | CPUs | Memory | GPUs |
|-----|------|--------|------|
| dev | 8–32 | 32–64Gi | — (add in pod.yaml if needed) |
| preprocess-dataset/ik | 180 × 1 | 180 × 2Gi | — |
| preprocess-dataset/moco-track | 180 × 10 | 180 × 16Gi | — |
| preprocess-dataset/normalization | 1 | 2Gi | — |
| preprocess-dataset/fit-function-paths/* | see Path fit section | — | — |
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

## Path fit (Job 2)

**Mode A (local):** `python scripts/fit_rajagopal_function_paths.py --sample_motions 200` — single process (`--phase all`), optional `--num_workers` / `--num_threads`.

**Mode C (cluster):** one entry point submits three work Jobs in order (orchestrator uses `kubectl wait`, no `sleep`):

```bash
kubectl delete job sindyffuse-path-fit-orchestrator --ignore-not-found -n YOUR_NAMESPACE
kubectl apply -k deploy/jobs/preprocess-dataset/fit-function-paths/orchestrator -n YOUR_NAMESPACE
```

Or from a laptop: `./deploy/scripts/run-path-fit.sh YOUR_NAMESPACE`

| Job | Resources | Role |
|-----|-----------|------|
| `sindyffuse-path-fit-orchestrator` | 1 CPU, 2Gi | apply + wait for work Jobs |
| `sindyffuse-path-fit-prepare` | 1 CPU, 2Gi | sample motions → manifest |
| `sindyffuse-path-fit-convert` | 180 × (1 CPU, 2Gi) | IndexedJob B3D → `.mot` |
| `sindyffuse-path-fit-fit` | 32 CPU, 32Gi | merge + OpenSim path fit |

Run after IK (Job 1) and before MocoTrack (Job 3).

## Pipeline order

1. `preprocess-dataset/ik` (or full orchestrator)
2. `preprocess-dataset/fit-function-paths/orchestrator` (or local Mode A script)
3. `preprocess-dataset/moco-track/orchestrator` (moco workers → normalization)
4. `train-sindy`
5. `train-surrogate`
6. `train-diffusion/{none,nimble,sindy}`

Single entry point for the full preprocess pipeline:

```bash
kubectl apply -k deploy/jobs/preprocess-dataset/orchestrator -n YOUR_NAMESPACE
```

## Results layout

Each trainer writes timestamped runs under ``results/<family>/runs/`` and updates ``results/<family>/latest`` when training completes (see ``common/run_setup.py``).

## Local development

No Kubernetes required for development:

```bash
conda env create -f env/environment.yaml
conda activate sindyffuse
python scripts/preprocess_ik.py --max_motions 1
python scripts/preprocess_moco.py --max_motions 1
```

See the root [README.md](../README.md) for the full local pipeline.
