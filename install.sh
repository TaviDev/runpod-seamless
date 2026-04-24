#!/usr/bin/env bash
# Seamless Translation PoC — install script
# Target: runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04 on a 4090.
#
# FRESH POD BOOTSTRAP:
#   cd /workspace
#   git clone https://github.com/TaviDev/runpod-seamless.git .scripts
#   bash .scripts/install.sh
#   # ...when install finishes...
#   ./start.sh m4t <audio.wav> --task S2TT --tgt_lang spa --model_name seamlessM4T_v2_large
#
# Idempotent: safe to re-run on a partially-set-up volume. Each section has
# a guard that skips the work if it's already been done.
#
# What ends up on the volume (~/workspace):
#   envs/seamless/                      Python venv (~14 GB)
#   seamless_communication/             cloned repo, installed as -e . (~200 MB)
#   models/fairseq2-cache/              populated lazily by fairseq2 on first inference
#   models/hf-cache/                    HF CLI config (token, etc.) — small
#   .scripts/                           this repo (for redeploy reproducibility)
#   install.sh, start.sh                convenience copies of the scripts

set -euo pipefail

WORKSPACE=/workspace
REPO_DIR=$WORKSPACE/seamless_communication
VENV_DIR=$WORKSPACE/envs/seamless
MODELS_DIR=$WORKSPACE/models
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

log() { echo -e "\n\033[1;36m[install] $*\033[0m"; }

# ---------------------------------------------------------------------------
# Section 0: Directory skeleton on the volume
# ---------------------------------------------------------------------------
create_dirs() {
  log "Section 0: volume directory skeleton"
  mkdir -p "$WORKSPACE/envs"
  mkdir -p "$MODELS_DIR/fairseq2-cache"
  mkdir -p "$MODELS_DIR/hf-cache"
}

# ---------------------------------------------------------------------------
# Section 1: System apt deps (container overlay — re-runs each fresh pod)
# ---------------------------------------------------------------------------
install_apt_deps() {
  log "Section 1: apt deps"
  if command -v ffmpeg >/dev/null \
     && command -v node >/dev/null \
     && command -v git-lfs >/dev/null \
     && ldconfig -p | grep -q libsndfile.so.1; then
    log "apt deps already present, skipping"
    return 0
  fi
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git-lfs build-essential python3-venv curl ca-certificates
  if ! command -v node >/dev/null || [[ "$(node --version)" != v20.* ]]; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
  fi
  git lfs install
}

# ---------------------------------------------------------------------------
# Section 2: Clone seamless_communication
# Pinned against commit 90e2b57 (main as of 2026-04-24). setup.py pins
# fairseq2==0.2.*, which drives all our Section 3 version choices.
# ---------------------------------------------------------------------------
clone_repo() {
  log "Section 2: clone seamless_communication"
  if [[ -d "$REPO_DIR/.git" ]]; then
    log "repo already present at $REPO_DIR, skipping"
    return 0
  fi
  git clone https://github.com/facebookresearch/seamless_communication.git "$REPO_DIR"
}

# ---------------------------------------------------------------------------
# Section 3: Python venv with torch + fairseq2 + numpy, atomically.
# THE dangerous section. The rules:
#  - One pip install call (single resolver pass).
#  - torch 2.1.1 + cu121 because fairseq2n 0.2.0 strictly pins torch==2.1.1.
#  - fairseq2 0.2.0 from FAIR's pt2.1.1/cu121 index (gives +cu121 variant).
#  - wheel<0.45 pinned preemptively to avoid sacrebleu's packaging<24 warning.
#  - numpy<2 locked because torch 2.1 extensions are not numpy-2 compatible.
# ---------------------------------------------------------------------------
install_python_stack() {
  log "Section 3: venv + torch + fairseq2 atomic install"
  if [[ -d "$VENV_DIR" ]]; then
    if "$VENV_DIR/bin/python" -c "from fairseq2.data.audio import AudioDecoder" 2>/dev/null; then
      log "venv already installed and ABI-healthy, skipping"
      return 0
    else
      log "venv exists but is broken, removing"
      rm -rf "$VENV_DIR"
    fi
  fi
  python3 -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  pip install --upgrade "pip==24.0" "setuptools" "wheel<0.45"
  pip install \
    --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.1.1/cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.1.1" "torchaudio==2.1.1" "fairseq2==0.2.0" "numpy<2" "wheel<0.45"
  python -c "
import torch
from fairseq2.data.audio import AudioDecoder
assert torch.cuda.is_available(), 'CUDA not available in venv'
x = torch.randn(100, 100, device='cuda') @ torch.randn(100, 100, device='cuda')
torch.cuda.synchronize()
print('Section 3 ABI check passed')
"
}

# ---------------------------------------------------------------------------
# Section 4: seamless_communication + its non-torch/non-fairseq deps.
# --no-deps on seamless itself so pip doesn't re-resolve torch/fairseq2.
# ---------------------------------------------------------------------------
install_seamless() {
  log "Section 4: seamless_communication package + deps"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  if python -c "from seamless_communication.inference import Translator" 2>/dev/null; then
    log "seamless_communication already importable, skipping"
    return 0
  fi
  cd "$REPO_DIR"
  pip install --no-deps -e .
  pip install \
    "datasets==2.18.0" "fire" "librosa" "openai-whisper" "simuleval~=1.1.3" \
    "sonar-space==0.2.*" "soundfile" "scipy" "tqdm" "numpy<2" "wheel<0.45"
  python -c "
import torch
from fairseq2.data.audio import AudioDecoder
from seamless_communication.inference import Translator
assert torch.__version__.startswith('2.1.1'), f'torch drift: {torch.__version__}'
assert torch.cuda.is_available()
print('Section 4 check passed')
"
}

# ---------------------------------------------------------------------------
# Section 5: Publish the scripts to convenient paths on the volume.
# The git clone landed them under $WORKSPACE/.scripts/ — copy install.sh and
# start.sh to $WORKSPACE/ too, so the usage line in the welcome message works.
# Models are NOT downloaded here; fairseq2 pulls them on first inference and
# caches to $FAIRSEQ2_CACHE_DIR (set by start.sh).
# ---------------------------------------------------------------------------
publish_scripts() {
  log "Section 5: copy scripts to /workspace/ for convenience"
  cp "$SCRIPT_DIR/install.sh" "$WORKSPACE/install.sh"
  cp "$SCRIPT_DIR/start.sh"   "$WORKSPACE/start.sh"
  chmod +x "$WORKSPACE/install.sh" "$WORKSPACE/start.sh"
}

main() {
  create_dirs
  install_apt_deps
  clone_repo
  install_python_stack
  install_seamless
  publish_scripts

  log "Install complete."
  log "Try: /workspace/start.sh m4t /workspace/seamless_communication/demo/dino_pretssel/audios/employee_eng_spa/ref/clean_spk1_default_00240.wav --task S2TT --tgt_lang spa --model_name seamlessM4T_v2_large"
}

main "$@"