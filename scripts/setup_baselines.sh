#!/usr/bin/env bash
# Clone baseline repos into baselines/ and download pre-trained checkpoints.
# Baseline code and weights are gitignored; this script is the only setup path.
#
# Usage:
#   ./scripts/setup_baselines.sh            # clone + download models
#   ./scripts/setup_baselines.sh --clone-only
#   ./scripts/setup_baselines.sh --models-only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${SINDYFFUSE_BASELINES_DIR:-$ROOT/baselines}"
MANIFEST="$ROOT/configs/baselines.json"
MARKER=".sindyffuse_setup_done"

CLONE_ONLY=0
MODELS_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --clone-only) CLONE_ONLY=1 ;;
    --models-only) MODELS_ONLY=1 ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

log() { echo "[setup_baselines] $*"; }
die() { echo "[setup_baselines] ERROR: $*" >&2; exit 1; }

ensure_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

ensure_gdown() {
  if command -v gdown >/dev/null 2>&1; then
    return 0
  fi
  log "Installing gdown (needed for checkpoint downloads)"
  python3 -m pip install -q gdown
  command -v gdown >/dev/null 2>&1 || die "Failed to install gdown"
}

run_in_repo() {
  local repo_dir="$1"
  shift
  (cd "$BASE/$repo_dir" && "$@")
}

run_script_if_exists() {
  local repo_dir="$1"
  local script="$2"
  if [[ -f "$BASE/$repo_dir/$script" ]]; then
    log "$repo_dir: running $script"
    run_in_repo "$repo_dir" bash "$script"
  else
    log "$repo_dir: skip missing $script"
  fi
}

download_gdrive_file() {
  local repo_dir="$1"
  local file_id="$2"
  local dest_name="$3"
  local dest="$BASE/$repo_dir/$dest_name"
  if [[ -e "$dest" ]]; then
    log "$repo_dir: already have $dest_name"
    return 0
  fi
  log "$repo_dir: downloading $dest_name"
  run_in_repo "$repo_dir" gdown "https://drive.google.com/uc?id=$file_id" -O "$dest_name"
}

unzip_if_needed() {
  local repo_dir="$1"
  local archive="$2"
  local dest_dir="${3:-.}"
  local path="$BASE/$repo_dir/$archive"
  [[ -f "$path" ]] || return 0
  log "$repo_dir: extracting $archive"
  run_in_repo "$repo_dir" unzip -o "$archive" -d "$dest_dir"
  run_in_repo "$repo_dir" rm -f "$archive"
}

clone_repos() {
  python3 - "$MANIFEST" "$BASE" <<'PY'
import json, subprocess, sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
base = Path(sys.argv[2])
for repo in manifest["repos"]:
    dest = base / repo["dir"]
    url = repo["url"]
    if (dest / ".git").is_dir():
        print(f"[setup_baselines] [skip] {repo['dir']} already cloned")
    else:
        print(f"[setup_baselines] [clone] {url} -> {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
    (dest / "outputs").mkdir(parents=True, exist_ok=True)
PY
}

setup_mdm() {
  local repo="motion-diffusion-model"
  [[ -d "$BASE/$repo/.git" ]] || return 0
  run_script_if_exists "$repo" "prepare/download_glove.sh"
  run_script_if_exists "$repo" "prepare/download_t2m_evaluators.sh"
  mkdir -p "$BASE/$repo/save"
  download_gdrive_file "$repo" "1cfadR1eZ116TIdXK7qDX1RugAerEiJXr" "humanml_encoder_512_50steps.zip"
  unzip_if_needed "$repo" "humanml_encoder_512_50steps.zip" "save"
}

setup_mld() {
  local repo="motion-latent-diffusion"
  [[ -d "$BASE/$repo/.git" ]] || return 0
  run_script_if_exists "$repo" "prepare/download_smpl_model.sh"
  run_script_if_exists "$repo" "prepare/prepare_clip.sh"
  run_script_if_exists "$repo" "prepare/download_t2m_evaluators.sh"
  run_script_if_exists "$repo" "prepare/download_pretrained_models.sh"
}

setup_t2m_gpt() {
  local repo="T2M-GPT"
  [[ -d "$BASE/$repo/.git" ]] || return 0
  run_script_if_exists "$repo" "dataset/prepare/download_glove.sh"
  run_script_if_exists "$repo" "dataset/prepare/download_extractor.sh"
  run_script_if_exists "$repo" "dataset/prepare/download_model.sh"
}

setup_t2m() {
  local repo="text-to-motion"
  [[ -d "$BASE/$repo/.git" ]] || return 0
  mkdir -p "$BASE/$repo/checkpoints"
  download_gdrive_file "$repo" "1IgrFCnxeg4olBtURUHimzS03ZI0df_6W" "t2m_pretrained.zip"
  unzip_if_needed "$repo" "t2m_pretrained.zip" "checkpoints"
}

setup_motiondiffuse() {
  local repo="MotionDiffuse"
  [[ -d "$BASE/$repo/.git" ]] || return 0
  if [[ -d "$BASE/$repo/checkpoints/t2m/t2m_motiondiffuse" ]]; then
    log "$repo: checkpoints already present"
    return 0
  fi
  log "$repo: downloading shared MotionDiffuse/ReMoDiffuse data bundle"
  run_in_repo "$repo" gdown --folder "https://drive.google.com/drive/folders/13kwahiktQ2GMVKfVH3WT-VGAQ6JHbvUv" -O _downloads || true
  if [[ -d "$BASE/$repo/_downloads" ]]; then
    run_in_repo "$repo" bash -c 'shopt -s dotglob; for item in _downloads/*; do base=$(basename "$item"); if [[ "$base" == checkpoints || "$base" == data ]]; then cp -rn "$item" .; fi; done'
    run_in_repo "$repo" rm -rf _downloads
  fi
}

setup_remodiffuse() {
  local repo="ReMoDiffuse"
  [[ -d "$BASE/$repo/.git" ]] || return 0
  if [[ -d "$BASE/$repo/logs/remodiffuse" || -d "$BASE/$repo/data/evaluators" ]]; then
    log "$repo: data bundle already present"
    return 0
  fi
  log "$repo: downloading shared MotionDiffuse/ReMoDiffuse data bundle"
  run_in_repo "$repo" gdown --folder "https://drive.google.com/drive/folders/13kwahiktQ2GMVKfVH3WT-VGAQ6JHbvUv" -O _downloads || true
  if [[ -d "$BASE/$repo/_downloads" ]]; then
    run_in_repo "$repo" bash -c 'shopt -s dotglob; for item in _downloads/*; do cp -rn "$item" .; done'
    run_in_repo "$repo" rm -rf _downloads
  fi
}

setup_stablemofusion() {
  local repo="StableMoFusion"
  [[ -d "$BASE/$repo/.git" ]] || return 0
  if [[ -f "$BASE/$repo/checkpoints/t2m/t2m_condunet1d_batch64/model/latest.tar" ]]; then
    log "$repo: checkpoints already present"
    return 0
  fi
  log "$repo: downloading pre-trained checkpoints"
  mkdir -p "$BASE/$repo/checkpoints"
  run_in_repo "$repo" gdown --folder "https://drive.google.com/drive/folders/1o3h0DHEz5gKG-9cTdl3lUEwjwW51Ay81" -O _downloads || true
  if [[ -d "$BASE/$repo/_downloads" ]]; then
    run_in_repo "$repo" bash -c 'shopt -s dotglob; cp -rn _downloads/* checkpoints/ 2>/dev/null || cp -rn _downloads/* .'
    run_in_repo "$repo" rm -rf _downloads
  fi
}

mark_done() {
  local repo="$1"
  date -u +"%Y-%m-%dT%H:%M:%SZ" >"$BASE/$repo/$MARKER"
}

download_models() {
  ensure_gdown
  ensure_cmd unzip

  setup_mdm
  setup_mld
  setup_t2m_gpt
  setup_t2m
  setup_motiondiffuse
  setup_remodiffuse
  setup_stablemofusion

  python3 - "$MANIFEST" "$BASE" "$MARKER" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
base = Path(sys.argv[2])
marker = sys.argv[3]
stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
for repo in manifest["repos"]:
    path = base / repo["dir"] / marker
    if (base / repo["dir"] / ".git").is_dir():
        path.write_text(stamp + "\n")
PY
}

main() {
  [[ -f "$MANIFEST" ]] || die "Missing manifest: $MANIFEST"
  ensure_cmd git
  ensure_cmd python3
  mkdir -p "$BASE"

  if [[ "$MODELS_ONLY" -eq 0 ]]; then
    clone_repos
  fi

  if [[ "$CLONE_ONLY" -eq 0 ]]; then
    download_models
  fi

  log "Done. Baselines live under $BASE (gitignored)."
  log "Write eval outputs to baselines/<repo>/outputs/ before running baseline_eval.py"
}

main "$@"
