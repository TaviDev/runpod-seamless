#!/usr/bin/env bash
# Streaming demo launcher — Starlette + Socket.IO backend on :7860,
# serving the built React frontend from ../streaming-react-app/dist/.
# Caller (start.sh demo-streaming) is responsible for venv activation,
# FAIRSEQ2_CACHE_DIR/HF_HOME exports, ensure_demo_deps, check_expressive_weights,
# and check_streaming_models (which also bridges pretssel_melhifigan_wm.pt).

set -euo pipefail

CHOICE=${1:-expressive}
case "$CHOICE" in
  expressive)
    export USE_EXPRESSIVE_MODEL=1
    ;;
  non-expressive)
    export USE_EXPRESSIVE_MODEL=0
    ;;
  *)
    echo "Usage: $0 {expressive|non-expressive}" >&2
    exit 1
    ;;
esac

SERVER_DIR=/workspace/seamless-streaming-demo/seamless_server
DIST_DIR=/workspace/seamless-streaming-demo/streaming-react-app/dist

if [[ ! -d "$SERVER_DIR" ]]; then
  echo "ERROR: $SERVER_DIR missing — clone the HF Space first" >&2
  exit 1
fi
if [[ ! -d "$DIST_DIR" ]]; then
  echo "ERROR: React build missing — run yarn build in streaming-react-app/" >&2
  exit 1
fi

cd "$SERVER_DIR"
echo "[launch_streaming] $CHOICE on :7860  (USE_EXPRESSIVE_MODEL=$USE_EXPRESSIVE_MODEL)"
exec uvicorn app_pubsub:app --host 0.0.0.0 --port 7860
