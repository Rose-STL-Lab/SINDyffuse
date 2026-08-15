#!/usr/bin/env bash
# Build + optionally push the SINDyffuse runtime image (Docker Hub: ncking/sindyffuse).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
IMAGE="${IMAGE:-ncking/sindyffuse:latest}"
DOCKER="${DOCKER:-}"

pick_docker() {
  if [[ -n "${DOCKER}" ]]; then
    return 0
  fi
  if docker info >/dev/null 2>&1; then
    DOCKER="docker"
    return 0
  fi
  if command -v sudo >/dev/null && sudo -n docker info >/dev/null 2>&1; then
    DOCKER="sudo docker"
    return 0
  fi
  if command -v sudo >/dev/null && sudo docker info >/dev/null 2>&1; then
    DOCKER="sudo docker"
    return 0
  fi
  echo "Docker daemon not reachable. Start it: sudo systemctl start docker" >&2
  echo "Then: sudo usermod -aG docker \"\$USER\" && newgrp docker" >&2
  echo "Or: sudo docker build -f env/Dockerfile -t ${IMAGE} ." >&2
  exit 1
}

pick_docker
echo "Building $IMAGE with: $DOCKER"
$DOCKER build -f env/Dockerfile -t "$IMAGE" .
echo "Build OK. Smoke:"
$DOCKER run --rm "$IMAGE" python -c "
import numpy as np
import opensim
import casadi
assert np.__version__.startswith('1.25'), np.__version__
assert hasattr(opensim, 'Model')
print('numpy', np.__version__)
print('opensim', opensim.GetVersion())
print('casadi', casadi.__version__)
print('opensim.moco (same process as casadi)', hasattr(opensim, 'MocoStudy'))
"
# Moco A/B path: fresh process, no casadi import first.
$DOCKER run --rm "$IMAGE" python -c "
import opensim
assert hasattr(opensim, 'MocoStudy'), 'OpenSim Moco missing'
print('opensim.moco (solo process)', True)
"
if [[ "${PUSH:-0}" == "1" ]]; then
  $DOCKER push "$IMAGE"
  echo "Pushed $IMAGE"
fi
