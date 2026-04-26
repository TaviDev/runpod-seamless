#!/usr/bin/env bash
# Offline Gradio demo launcher — m4tv2 (port 7860) or expressive (port 7861).
# Caller (start.sh demo-offline) is responsible for venv activation,
# CHECKPOINTS_PATH/FAIRSEQ2_CACHE_DIR/HF_HOME exports, and ensure_demo_deps.

set -euo pipefail

CHOICE=${1:-}
case "$CHOICE" in
  m4t)
    APP_DIR=/workspace/seamless_communication/demo/m4tv2
    PORT=7860
    ;;
  expressive)
    APP_DIR=/workspace/seamless_communication/demo/expressive
    PORT=7862
    ;;
  *)
    echo "Usage: $0 {m4t|expressive}" >&2
    exit 1
    ;;
esac

if [[ ! -f "$APP_DIR/app.py" ]]; then
  echo "ERROR: app.py not found at $APP_DIR" >&2
  exit 1
fi

export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=$PORT
# CHECKPOINTS_PATH is what the apps' InProcAssetMetadataProvider reads.
export CHECKPOINTS_PATH=${CHECKPOINTS_PATH:-/workspace/models/checkpoints-demo}

cd "$APP_DIR"
echo "[launch_offline] $CHOICE on :$PORT  (cwd=$APP_DIR  CHECKPOINTS_PATH=$CHECKPOINTS_PATH)"
exec python app.py
