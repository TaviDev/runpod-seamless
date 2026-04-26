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
# Env vars consumed:
#   HF_TOKEN   (optional) HuggingFace token for gated SeamlessExpressive
#              download. Set on the RunPod private template. If unset,
#              install completes but `start.sh expressive` will be unusable
#              until the model is downloaded manually.
#
# What ends up on the volume (~/workspace):
#   envs/seamless/                      Python venv (~14 GB)
#   seamless_communication/             cloned repo, installed as -e . (~200 MB)
#   seamless-streaming-demo/            cloned HF Space + built React dist/ (~500 MB)
#   models/fairseq2-cache/              populated lazily by fairseq2 on first inference
#   models/hf-cache/                    HF CLI config (token, etc.) — small
#   models/hf-gated/seamless-expressive/ SeamlessExpressive weights (~22 GB)
#   models/checkpoints-demo/            Gradio demo CHECKPOINTS_PATH target
#   models/.demo-deps-installed-v2      sentinel: demo HTTP stack pinned correctly
#   npm-global/                         volume-backed npm prefix — Claude Code CLI + yarn
#   .claude/                            Claude Code CLI config (CLAUDE_CONFIG_DIR)
#   .scripts/                           this repo (for redeploy reproducibility)
#   install.sh, start.sh                convenience copies of the scripts

set -euo pipefail

WORKSPACE=/workspace
REPO_DIR=$WORKSPACE/seamless_communication
VENV_DIR=$WORKSPACE/envs/seamless
MODELS_DIR=$WORKSPACE/models
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

log() { echo -e "\n\033[1;36m[install] $*\033[0m"; }

# Prompt for HF_TOKEN up-front if unset and stdin is a tty. Empty input
# (or non-interactive runs) fall through to the soft-skip in Section 6.
prompt_hf_token_if_unset() {
  if [[ -n "${HF_TOKEN:-}" ]]; then return 0; fi
  # Token cached on the volume by a prior `hf auth login` survives pod
  # restarts (HF_HOME lives on /workspace). huggingface_hub reads it
  # automatically — no env var needed.
  if [[ -s "$MODELS_DIR/hf-cache/token" ]]; then
    log "HF token cached on volume ($MODELS_DIR/hf-cache/token) — skipping prompt."
    return 0
  fi
  if [[ -f "$MODELS_DIR/hf-gated/seamless-expressive/pretssel_melhifigan_wm-final.pt" ]]; then
    log "SeamlessExpressive already on volume — skipping HF_TOKEN prompt."
    return 0
  fi
  if [[ ! -t 0 ]]; then
    log "HF_TOKEN unset and stdin is not a tty — gated downloads will be skipped."
    return 0
  fi
  echo
  echo "HF_TOKEN is not set. SeamlessExpressive (~22 GB) is HF-gated and"
  echo "won't be auto-downloaded without a token. Get one at:"
  echo "  https://huggingface.co/settings/tokens   (read access is enough)"
  echo "and accept the model terms at:"
  echo "  https://huggingface.co/facebook/seamless-expressive"
  echo
  read -r -s -p "Paste HF_TOKEN now (or press Enter to skip): " HF_TOKEN
  echo
  if [[ -n "$HF_TOKEN" ]]; then
    export HF_TOKEN
    log "HF_TOKEN captured for this run."
  else
    log "No token given — gated downloads will be deferred."
  fi
}

# ---------------------------------------------------------------------------
# Section 0: Directory skeleton on the volume
# ---------------------------------------------------------------------------
create_dirs() {
  log "Section 0: volume directory skeleton"
  mkdir -p "$WORKSPACE/envs"
  mkdir -p "$MODELS_DIR/fairseq2-cache"
  mkdir -p "$MODELS_DIR/hf-cache"
  mkdir -p "$MODELS_DIR/hf-gated/seamless-expressive"
  mkdir -p "$MODELS_DIR/checkpoints-demo"
}

# ---------------------------------------------------------------------------
# Section 1: System apt deps + node20 + Claude Code CLI (container overlay —
# re-runs each fresh pod). Also drops /etc/profile.d/seamless.sh so SSH
# sessions inherit the volume-backed npm prefix + CLAUDE_CONFIG_DIR without
# sourcing start.sh.
# ---------------------------------------------------------------------------
install_apt_deps() {
  log "Section 1: apt deps + node + claude CLI"

  if ! { command -v ffmpeg >/dev/null \
        && command -v git-lfs >/dev/null \
        && ldconfig -p | grep -q libsndfile.so.1; }; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
      libsndfile1 ffmpeg git-lfs build-essential python3-venv curl ca-certificates
    git lfs install
  else
    log "apt deps already present, skipping"
  fi

  if ! command -v node >/dev/null || [[ "$(node --version)" != v20.* ]]; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
  fi

  # yarn — needed by Section 5 to build the streaming-react-app frontend.
  # Lives under the volume-backed NPM_CONFIG_PREFIX (set below), so it
  # survives container redeploys.

  # /etc/profile.d/seamless.sh — sourced by login shells (SSH) so `claude`
  # resolves and reads its config from the volume even outside start.sh.
  cat > /etc/profile.d/seamless.sh <<'EOF'
# Seamless PoC — volume-backed paths for npm + claude
export NPM_CONFIG_PREFIX=/workspace/npm-global
export PATH=$NPM_CONFIG_PREFIX/bin:$PATH
export CLAUDE_CONFIG_DIR=/workspace/.claude
EOF
  chmod 0644 /etc/profile.d/seamless.sh

  # Claude Code CLI under the volume-backed npm prefix so it survives pod
  # redeploys. Re-installs (npm-global is wiped) are a no-op once present.
  export NPM_CONFIG_PREFIX=$WORKSPACE/npm-global
  export PATH=$NPM_CONFIG_PREFIX/bin:$PATH
  mkdir -p "$NPM_CONFIG_PREFIX"
  if ! command -v claude >/dev/null; then
    npm install -g @anthropic-ai/claude-code
  else
    log "claude CLI already present, skipping"
  fi

  if ! command -v yarn >/dev/null; then
    npm install -g yarn
  else
    log "yarn already present, skipping"
  fi
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
# Section 4: seamless_communication + its non-torch/non-fairseq deps + the
# offline-Gradio HTTP stack.
# --no-deps on seamless itself so pip doesn't re-resolve torch/fairseq2.
# starlette/fastapi/uvicorn pinned because pip's resolver otherwise pulls
# starlette 1.0.0, which breaks gradio 4.5's templates.TemplateResponse with
# `TypeError: unhashable type: 'dict'` in jinja2's LRUCache.
# ---------------------------------------------------------------------------
DEMO_DEPS_MARKER=$WORKSPACE/models/.demo-deps-installed-v2

install_seamless() {
  log "Section 4: seamless_communication package + deps"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  if [[ -f "$DEMO_DEPS_MARKER" ]] && python -c "
from seamless_communication.inference import Translator
import gradio, omegaconf, starlette, fastapi, uvicorn
" 2>/dev/null; then
    log "seamless_communication + demo deps already installed, skipping"
    return 0
  fi
  cd "$REPO_DIR"
  pip install --no-deps -e .
  pip install \
    "datasets==2.18.0" "fire" "librosa" "openai-whisper" "simuleval~=1.1.3" \
    "sonar-space==0.2.*" "soundfile" "scipy" "tqdm" "numpy<2" "wheel<0.45" \
    "huggingface_hub>=0.20" "gradio~=4.5.0" "gradio_client" "omegaconf~=2.3.0" \
    "starlette>=0.27,<0.40" "fastapi>=0.104,<0.110" "uvicorn>=0.23,<0.35"
  python -c "
import torch
from fairseq2.data.audio import AudioDecoder
from seamless_communication.inference import Translator
import gradio, omegaconf, starlette, fastapi, uvicorn
assert torch.__version__.startswith('2.1.1'), f'torch drift: {torch.__version__}'
assert torch.cuda.is_available()
print('Section 4 check passed')
"
  touch "$DEMO_DEPS_MARKER"
}

# ---------------------------------------------------------------------------
# Section 5: Streaming demo bootstrap (HF Space clone + Python deps + React
# build). Skipped from upstream `requirements.txt`: Flask/Flask_Sockets/gevent/
# Werkzeug (vestigial — `app_pubsub` is Starlette), numpy==1.24.4 (we keep
# 1.26.4), whisper==1.1.10 (different package; not what `app_pubsub` imports),
# openai_whisper==20230124 (we keep 20250625, API-stable).
# All --no-deps because pip's resolver would otherwise re-pull numpy 2 / a
# different starlette / etc. and shred Section 3+4. Leaf deps below were
# discovered empirically by ImportError in Session 5 (python-socketio needs
# python-engineio+bidict+simple-websocket; python-jose needs ecdsa+pyasn1+rsa;
# g2p_en needs nltk+inflect+distance; psola needs pypar+praat-parselmouth).
# ---------------------------------------------------------------------------
STREAMING_DIR=$WORKSPACE/seamless-streaming-demo

install_streaming_demo() {
  log "Section 5: streaming demo (HF Space + Python deps + React build)"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  if [[ ! -d "$STREAMING_DIR/.git" ]]; then
    log "  cloning facebook/seamless-streaming -> $STREAMING_DIR"
    git clone https://huggingface.co/spaces/facebook/seamless-streaming "$STREAMING_DIR"
  else
    log "  HF Space already cloned, skipping"
  fi

  if python -c "
import socketio, jose, g2p_en, psola, parselmouth, stable_whisper
import nltk, inflect, distance, pypar, ecdsa, pyasn1, rsa
import bidict, engineio, simple_websocket, wsproto
import silero, parallel_wavegan, colorlog, pydub, hf_transfer
" 2>/dev/null; then
    log "  streaming Python deps already present, skipping"
  else
    log "  installing streaming Python deps (--no-deps)"
    pip install --no-deps \
      "python-socketio==5.9.0" "python-engineio==4.7.1" \
      "bidict==0.22.1" "simple-websocket==1.0.0" "wsproto==1.2.0" \
      "python-jose[cryptography]==3.3.0" \
      "ecdsa==0.18.0" "pyasn1==0.5.0" "rsa==4.9" \
      "g2p_en==2.1.0" "nltk==3.8.1" "inflect==7.0.0" "distance==0.1.3" \
      "silero==0.4.1" "parallel-wavegan==0.5.5" "colorlog==6.7.0" "pydub==0.25.1" \
      "psola==0.0.1" "pypar==0.0.6" "praat-parselmouth==0.4.3" \
      "stable-ts==1.4.0" "hf_transfer==0.1.4"
    python -c "
import socketio, jose, g2p_en, psola, parselmouth, stable_whisper
print('Section 5 deps check passed')
"
  fi

  REACT_DIR=$STREAMING_DIR/streaming-react-app
  if [[ -d "$REACT_DIR/dist" ]]; then
    log "  React dist/ already built, skipping"
  else
    log "  building React frontend (yarn install + yarn build)"
    # shellcheck disable=SC2034
    export NPM_CONFIG_PREFIX=$WORKSPACE/npm-global
    export PATH=$NPM_CONFIG_PREFIX/bin:$PATH
    cd "$REACT_DIR"
    yarn install
    yarn build
  fi
}

# ---------------------------------------------------------------------------
# Section 6: HF auth + SeamlessExpressive (gated) download.
# fairseq2 lazily pulls non-gated models on first inference, but
# SeamlessExpressive is HF-gated and must be fetched via `hf download`. We
# bootstrap the HF token from the HF_TOKEN env var (set on the RunPod
# private template) so cold-volume rebuilds finish without manual steps.
# Soft-skip if HF_TOKEN is unset — install still completes, M4T flows work.
# ---------------------------------------------------------------------------
download_gated_models() {
  log "Section 6: HF auth + SeamlessExpressive"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  export HF_HOME="$MODELS_DIR/hf-cache"
  export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

  # Auth precedence: HF_TOKEN env var (fresh credential, e.g. first install or
  # rotation) → cached token at $HF_HOME/token (volume-persistent, survives
  # pod stop/start). hf/huggingface_hub auto-read the cached token, so we
  # only call `hf auth login` when given a new one.
  if [[ -n "${HF_TOKEN:-}" ]]; then
    hf auth login --token "$HF_TOKEN" --add-to-git-credential
  elif [[ -s "$HF_HOME/token" ]] && hf auth whoami >/dev/null 2>&1; then
    log "using cached HF token at $HF_HOME/token ($(hf auth whoami 2>/dev/null | head -1))"
  else
    log "no HF_TOKEN env var and no valid cached token — skipping gated downloads."
    log "  Set HF_TOKEN in your RunPod template env vars and re-run install.sh,"
    log "  or download manually:"
    log "    hf download facebook/seamless-expressive --repo-type model \\"
    log "      --local-dir $MODELS_DIR/hf-gated/seamless-expressive"
    return 0
  fi

  EXPR_DIR=$MODELS_DIR/hf-gated/seamless-expressive
  # Sentinel: pretssel_melhifigan_wm-final.pt is the unique 24kHz vocoder
  # checkpoint shipped in the release. Adjust if Meta renames upstream.
  if [[ ! -f "$EXPR_DIR/pretssel_melhifigan_wm-final.pt" ]]; then
    log "downloading SeamlessExpressive (~22 GB) -> $EXPR_DIR"
    hf download facebook/seamless-expressive --repo-type model --local-dir "$EXPR_DIR"
  else
    log "SeamlessExpressive already present, skipping"
  fi

  # The 24kHz vocoder card hardcodes the basename pretssel_melhifigan_wm.pt
  # but the release ships it as pretssel_melhifigan_wm-final.pt.
  if [[ ! -e "$EXPR_DIR/pretssel_melhifigan_wm.pt" \
        && -f "$EXPR_DIR/pretssel_melhifigan_wm-final.pt" ]]; then
    ln -sf pretssel_melhifigan_wm-final.pt "$EXPR_DIR/pretssel_melhifigan_wm.pt"
  fi
}

# ---------------------------------------------------------------------------
# Section 7: Publish the scripts to convenient paths on the volume.
# The git clone landed them under $WORKSPACE/.scripts/ — copy install.sh and
# start.sh to $WORKSPACE/ too, so the usage line in the welcome message works.
# Non-gated models are NOT downloaded here; fairseq2 pulls them on first
# inference and caches to $FAIRSEQ2_CACHE_DIR (set by start.sh).
# ---------------------------------------------------------------------------
publish_scripts() {
  log "Section 7: copy scripts to /workspace/ for convenience"
  cp "$SCRIPT_DIR/install.sh" "$WORKSPACE/install.sh"
  cp "$SCRIPT_DIR/start.sh"   "$WORKSPACE/start.sh"
  chmod +x "$WORKSPACE/install.sh" "$WORKSPACE/start.sh"
}

main() {
  prompt_hf_token_if_unset
  create_dirs
  install_apt_deps
  clone_repo
  install_python_stack
  install_seamless
  install_streaming_demo
  download_gated_models
  publish_scripts

  log "Install complete."
  log "Try: /workspace/start.sh m4t /workspace/seamless_communication/demo/dino_pretssel/audios/employee_eng_spa/ref/clean_spk1_default_00240.wav --task S2TT --tgt_lang spa --model_name seamlessM4T_v2_large"
}

main "$@"