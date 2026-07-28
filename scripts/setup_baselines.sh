#!/usr/bin/env bash
# Clone baseline text-to-motion repos into baselines/ and prepare output dirs.
#
# Pre-trained checkpoints are downloaded separately inside each repo; see each
# project's README after cloning.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${SINDYFFUSE_BASELINES_DIR:-$ROOT/baselines}"

clone_if_missing() {
  local dir="$1"
  local url="$2"
  if [[ -d "$BASE/$dir/.git" ]]; then
    echo "[skip] $dir already cloned"
  else
    echo "[clone] $url -> $BASE/$dir"
    git clone --depth 1 "$url" "$BASE/$dir"
  fi
  mkdir -p "$BASE/$dir/outputs"
}

mkdir -p "$BASE"

clone_if_missing motion-diffusion-model   https://github.com/GuyTevet/motion-diffusion-model.git
clone_if_missing motion-latent-diffusion  https://github.com/ChenFengYe/motion-latent-diffusion.git
clone_if_missing T2M-GPT                  https://github.com/Mael-zys/T2M-GPT.git
clone_if_missing text-to-motion           https://github.com/EricGuo5513/text-to-motion.git
clone_if_missing MotionDiffuse            https://github.com/mingyuan-zhang/MotionDiffuse.git
clone_if_missing ReMoDiffuse              https://github.com/mingyuan-zhang/ReMoDiffuse.git
clone_if_missing StableMoFusion           https://github.com/Linketic/StableMoFusion.git

echo "Baselines ready under $BASE"
echo "Download pre-trained weights per repo README, then write eval motions to baselines/<repo>/outputs/"
