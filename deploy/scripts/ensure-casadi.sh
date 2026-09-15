#!/usr/bin/env bash
# Source after job-env.sh. Repair OpenSimAD runtime deps for incomplete images/envs.
# CasADi: env/Dockerfile removes conda libcasadi and pins pip casadi==3.7.1.
# seaborn/pyyaml: listed in environment.yaml but often missing from partial envs;
# utilsOpenSimAD / mainOpenSimAD import them at module load.
#
# CRITICAL: OpenSim 4.5.2 requires numpy 1.25.x. Install seaborn/pyyaml with deps
# under env/constraints.txt (PIP_CONSTRAINT) so matplotlib is pulled without
# upgrading numpy to 2.x. A bare --no-deps install fails when matplotlib is absent.
set -eo pipefail

: "${CONDA_PREFIX:?source deploy/scripts/job-env.sh first (conda activate sindyffuse)}"

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_PATCH="${_REPO_ROOT}/env/patch_opensim_moco.py"
_CONSTRAINTS="${_REPO_ROOT}/env/constraints.txt"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mnt/SINDyffuse/.cache/pip}"
mkdir -p "${PIP_CACHE_DIR}" 2>/dev/null || true
if [[ -f "${_CONSTRAINTS}" ]]; then
  export PIP_CONSTRAINT="${PIP_CONSTRAINT:-${_CONSTRAINTS}}"
fi

_casadi_ok() {
  python -c "import casadi; assert casadi.__version__.startswith('3.7.1'), casadi.__version__" 2>/dev/null
}

_numpy_ok() {
  python -c "import numpy as np; assert np.__version__.startswith('1.25'), np.__version__" 2>/dev/null
}

_apply_moco_soft_import() {
  # Soft-import opensim.moco so pip CasADi can load in the same process (env/Dockerfile).
  if [[ -f "${_PATCH}" ]]; then
    python "${_PATCH}" || true
  fi
}

_ensure_numpy_125() {
  if _numpy_ok; then
    return 0
  fi
  echo "Restoring numpy 1.25.x (OpenSim requires it; got $(python -c 'import numpy as np; print(np.__version__)' 2>/dev/null || echo missing))"
  python -m pip install --force-reinstall 'numpy>=1.25,<1.26'
  _numpy_ok
}

_ensure_pip_mod() {
  # $1 = import name, $2 = pip requirement
  # Install with deps under PIP_CONSTRAINT (numpy>=1.25,<1.26) so incomplete
  # images get matplotlib/pandas without upgrading numpy to 2.x. --no-deps
  # alone fails when matplotlib is also missing (seaborn imports it at load).
  local import_name="$1"
  local requirement="$2"
  if python -c "import ${import_name}" 2>/dev/null; then
    return 0
  fi
  echo "Missing ${import_name}; installing ${requirement} (deps + PIP_CONSTRAINT)"
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
_ensure_numpy_125
