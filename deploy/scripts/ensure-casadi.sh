#!/usr/bin/env bash
# Source after job-env.sh. Repair OpenSimAD runtime deps for incomplete images/envs.
# CasADi: env/Dockerfile removes conda libcasadi and pins pip casadi==3.7.1.
# seaborn/pyyaml: listed in environment.yaml but often missing from partial envs;
# utilsOpenSimAD / mainOpenSimAD import them at module load.
set -eo pipefail

: "${CONDA_PREFIX:?source deploy/scripts/job-env.sh first (conda activate sindyffuse)}"

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_PATCH="${_REPO_ROOT}/env/patch_opensim_moco.py"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mnt/SINDyffuse/.cache/pip}"
mkdir -p "${PIP_CACHE_DIR}" 2>/dev/null || true

_casadi_ok() {
  python -c "import casadi; assert casadi.__version__.startswith('3.7.1'), casadi.__version__" 2>/dev/null
}

_apply_moco_soft_import() {
  # Soft-import opensim.moco so pip CasADi can load in the same process (env/Dockerfile).
  if [[ -f "${_PATCH}" ]]; then
    python "${_PATCH}" || true
  fi
}

_ensure_pip_mod() {
  # $1 = import name, $2 = pip requirement
  local import_name="$1"
  local requirement="$2"
  if python -c "import ${import_name}" 2>/dev/null; then
    return 0
  fi
  echo "Missing ${import_name}; installing ${requirement}"
  python -m pip install "${requirement}"
  python -c "import ${import_name}"
}

if _casadi_ok; then
  _apply_moco_soft_import
else
  echo "CasADi missing or wrong version; installing pip casadi==3.7.1 (OpenSimAD)"
  mamba remove -n sindyffuse -y --force casadi 2>/dev/null || true
  rm -f "${CONDA_PREFIX}/lib/libcasadi.so"* "${CONDA_PREFIX}/lib/libcasadi_"* || true
  rm -rf "${CONDA_PREFIX}/casadi" || true
  python -m pip install --no-deps --force-reinstall 'casadi==3.7.1'
  _casadi_ok
  _apply_moco_soft_import
  echo "casadi $(python -c 'import casadi; print(casadi.__version__)')"
fi

_ensure_pip_mod seaborn 'seaborn>=0.13'
_ensure_pip_mod yaml 'pyyaml>=6.0'
