#!/usr/bin/env bash
# Seamless Translation PoC — runtime entry point.
# Sets up env vars, activates the venv, dispatches to the requested mode.
#
# Usage:
#   ./start.sh m4t <audio.wav> --task {ASR,S2ST,S2TT} --tgt_lang <lang> [...]
#   ./start.sh streaming                 streaming server (TODO)
#   ./start.sh shell                     bash with env + venv active
#   ./start.sh python [args...]          python with env + venv active

set -euo pipefail

WORKSPACE=/workspace
VENV_DIR=$WORKSPACE/envs/seamless

# HF_HOME keeps the HF auth token on the volume (for future gated-model use).
export HF_HOME=$WORKSPACE/models/hf-cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

# FAIRSEQ2_CACHE_DIR is THE important one — this is where model weights land.
# fairseq2's download_manager.py checks this first, falling back to
# $XDG_CACHE_HOME then ~/.cache. Keeping it on the volume means models
# persist across pod redeploys.
export FAIRSEQ2_CACHE_DIR=$WORKSPACE/models/fairseq2-cache
mkdir -p "$FAIRSEQ2_CACHE_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found at $VENV_DIR"
  echo "Run /workspace/install.sh first (or bash /workspace/.scripts/install.sh)"
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

MODE=${1:-shell}
shift || true

case "$MODE" in
  m4t)
    exec m4t_predict "$@"
    ;;
  streaming)
    echo "streaming mode not yet wired up — coming in next phase"
    exit 1
    ;;
  shell)
    exec bash
    ;;
  python)
    exec python "$@"
    ;;
  *)
    echo "Usage: $0 {m4t|streaming|shell|python} [args...]"
    exit 1
    ;;
esac