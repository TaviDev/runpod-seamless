#!/usr/bin/env bash
# Seamless Translation PoC — runtime entry point.
# Sets up env vars, activates the venv, dispatches to the requested mode.
#
# Usage:
#   ./start.sh m4t <audio.wav> --task {ASR,S2ST,S2TT} --tgt_lang <lang> [...]
#   ./start.sh expressive <audio.wav> --tgt_lang <lang> [...]
#   ./start.sh try <audio.wav> [tgt_lang]    M4T S2TT + M4T S2ST + expressive S2ST, side-by-side
#   ./start.sh demo-offline [m4t|expressive|both]   browser Gradio demos on :7860/:7862 (default both)
#   ./start.sh demo-streaming [expressive|non-expressive]   real-time mic demo on :7860 (default expressive)
#   ./start.sh claude                    Claude Code CLI with volume-backed config
#   ./start.sh test-m4t                  smoke-test M4T v2 against reference WAV
#   ./start.sh test-expressive           smoke-test SeamlessExpressive against reference WAV
#   ./start.sh test-streaming            smoke-test streaming stack (import + asset resolution)
#   ./start.sh test-demo-offline         launch demo-offline both, probe :7860 + :7862, tear down
#   ./start.sh test-all                  run all three smoke tests, summarize
#   ./start.sh shell                     bash with env + venv active
#   ./start.sh python [args...]          python with env + venv active

set -euo pipefail

WORKSPACE=/workspace
VENV_DIR=$WORKSPACE/envs/seamless
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# HF_HOME keeps the HF auth token on the volume (for gated-model downloads).
export HF_HOME=$WORKSPACE/models/hf-cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

# FAIRSEQ2_CACHE_DIR is THE important one — this is where model weights land.
# fairseq2's download_manager.py checks this first, falling back to
# $XDG_CACHE_HOME then ~/.cache. Keeping it on the volume means models
# persist across pod redeploys.
export FAIRSEQ2_CACHE_DIR=$WORKSPACE/models/fairseq2-cache
mkdir -p "$FAIRSEQ2_CACHE_DIR"

# SeamlessExpressive is HF-gated and NOT fetched by fairseq2 — it's
# downloaded manually via `hf download` and passed via --gated-model-dir.
EXPRESSIVE_MODEL_DIR=$WORKSPACE/models/hf-gated/seamless-expressive

# Symlink farm consumed by the offline Gradio apps (both demos register assets
# via InProcAssetMetadataProvider with file://$CHECKPOINTS_PATH/<basename> URIs).
# Built/refreshed by demo/setup_checkpoints.py.
export CHECKPOINTS_PATH=$WORKSPACE/models/checkpoints-demo

# Volume-backed Claude Code config + npm global prefix. /etc/profile.d/seamless.sh
# (dropped by install.sh) sets these for SSH sessions; we re-export here so
# `start.sh claude` works even from a bare-bash invocation.
export NPM_CONFIG_PREFIX=$WORKSPACE/npm-global
export PATH=$NPM_CONFIG_PREFIX/bin:$PATH
export CLAUDE_CONFIG_DIR=$WORKSPACE/.claude

# Reference audio used by test modes. English source:
# "Do you want to resume the first, second or third timer?"
REF_WAV=$WORKSPACE/seamless_communication/demo/dino_pretssel/audios/employee_eng_spa/ref/clean_spk1_default_00240.wav

# Where test-mode outputs land (WAVs you can listen to afterwards).
TEST_OUT=$WORKSPACE/test-outputs
mkdir -p "$TEST_OUT"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found at $VENV_DIR"
  echo "Run /workspace/install.sh first (or bash /workspace/.scripts/install.sh)"
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

MODE=${1:-shell}
shift || true

# --- helpers ---

check_expressive_weights() {
  if [[ ! -d "$EXPRESSIVE_MODEL_DIR" ]] || [[ -z "$(ls -A "$EXPRESSIVE_MODEL_DIR" 2>/dev/null)" ]]; then
    echo "ERROR: SeamlessExpressive weights not found at $EXPRESSIVE_MODEL_DIR"
    echo "Download them first:"
    echo "  hf download facebook/seamless-expressive --repo-type model \\"
    echo "    --local-dir $EXPRESSIVE_MODEL_DIR"
    return 1
  fi
  # The 24kHz vocoder card hardcodes the basename `pretssel_melhifigan_wm.pt`,
  # but the HF release ships it as `pretssel_melhifigan_wm-final.pt`.
  # Make sure the symlink exists.
  if [[ ! -e "$EXPRESSIVE_MODEL_DIR/pretssel_melhifigan_wm.pt" ]]; then
    if [[ -f "$EXPRESSIVE_MODEL_DIR/pretssel_melhifigan_wm-final.pt" ]]; then
      ln -sf pretssel_melhifigan_wm-final.pt \
             "$EXPRESSIVE_MODEL_DIR/pretssel_melhifigan_wm.pt"
    else
      echo "ERROR: pretssel_melhifigan_wm-final.pt missing from $EXPRESSIVE_MODEL_DIR"
      return 1
    fi
  fi
}

# install.sh Section 4 lists gradio + gradio_client (and we need omegaconf for
# the expressive Gradio app), but its idempotency guard short-circuits the
# whole section once seamless_communication is importable. On volumes where
# Section 4 ran before those packages were added, we install them on demand.
ensure_demo_deps() {
  # Bumped to v2 after pinning starlette/fastapi/uvicorn — the earlier `pip
  # install gradio~=4.5.0` pulled in starlette 1.0.0 + fastapi 0.136.1, which
  # breaks gradio 4.5.0's templating (jinja2 cache key TypeError).
  local marker=$WORKSPACE/models/.demo-deps-installed-v2
  if [[ -f "$marker" ]] && python -c "import gradio, omegaconf, starlette, fastapi" 2>/dev/null; then
    return 0
  fi
  echo "[ensure_demo_deps] installing gradio + omegaconf + pinned HTTP stack (one-time)..."
  pip freeze > "$TEST_OUT/pip-freeze-pre-demo.txt"
  # Pin starlette/fastapi/uvicorn to the gradio 4.5.0-era window. Without
  # these, pip's resolver pulls in the latest, which breaks template render.
  pip install --upgrade-strategy only-if-needed \
    "gradio~=4.5.0" "gradio_client" "omegaconf~=2.3.0" "numpy<2" \
    "starlette>=0.27,<0.40" "fastapi>=0.104,<0.110" "uvicorn>=0.23,<0.35"
  python -c "
import torch, gradio, omegaconf, starlette, fastapi
assert torch.__version__.startswith('2.1.1'), f'torch drift: {torch.__version__}'
print(f'  gradio={gradio.__version__}  starlette={starlette.__version__}  fastapi={fastapi.__version__}')
" || {
    echo "ERROR: post-install ABI check failed — see $TEST_OUT/pip-freeze-pre-demo.txt"
    return 1
  }
  touch "$marker"
}

# The streaming HF Space's seamless_server expects models/Seamless/
# pretssel_melhifigan_wm.pt relative to its own dir. Bridge that to our
# canonical gated weights via a symlink. Idempotent.
check_streaming_models() {
  local server_dir=$WORKSPACE/seamless-streaming-demo/seamless_server
  if [[ ! -d "$server_dir" ]]; then
    echo "ERROR: streaming demo not cloned — expected $server_dir"
    echo "  git clone https://huggingface.co/spaces/facebook/seamless-streaming \\"
    echo "    $WORKSPACE/seamless-streaming-demo"
    return 1
  fi
  check_expressive_weights || return $?
  mkdir -p "$server_dir/models/Seamless"
  if [[ ! -e "$server_dir/models/Seamless/pretssel_melhifigan_wm.pt" ]]; then
    ln -sf "$EXPRESSIVE_MODEL_DIR/pretssel_melhifigan_wm.pt" \
           "$server_dir/models/Seamless/pretssel_melhifigan_wm.pt"
  fi
}

# Print the RunPod proxy URL for a given exposed port (best-effort).
print_proxy_url() {
  local port=$1
  local label=$2
  if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    echo "  $label  https://${RUNPOD_POD_ID}-${port}.proxy.runpod.net/"
  else
    echo "  $label  http://0.0.0.0:${port}/   (set RUNPOD_POD_ID for proxy URL)"
  fi
}

# SIGTERM (then SIGKILL) every process holding a LISTEN socket on the given
# TCP ports. test-demo-offline cleanup needs this because launch_offline.sh
# exec's into `python app.py` — the wrapper path drops out of the proc title,
# so `pkill -f launch_offline.sh` misses the live python. Tracking the PID
# from the listening socket is the only reliable handle without a pidfile.
kill_listeners_on() {
  local pids=() pid line
  for port in "$@"; do
    while read -r pid; do [[ -n "$pid" ]] && pids+=("$pid"); done < <(
      ss -ltnp 2>/dev/null | awk -v p="$port" '
        $4 ~ ":"p"$" {
          line = $0
          while (match(line, /pid=[0-9]+/)) {
            print substr(line, RSTART+4, RLENGTH-4)
            line = substr(line, RSTART+RLENGTH)
          }
        }'
    )
  done
  if (( ${#pids[@]} > 0 )); then
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep 3
    kill -KILL "${pids[@]}" 2>/dev/null || true
  fi
}

run_test_m4t() {
  echo "=== test-m4t: M4T v2 large, S2TT eng->spa ==="
  m4t_predict "$REF_WAV" \
    --task S2TT \
    --tgt_lang spa \
    --model_name seamlessM4T_v2_large
}

run_test_expressive() {
  echo "=== test-expressive: SeamlessExpressive, S2ST eng->spa ==="
  check_expressive_weights || return $?
  expressivity_predict "$REF_WAV" \
    --task S2ST \
    --src_lang eng \
    --tgt_lang spa \
    --vocoder_name vocoder_pretssel \
    --gated-model-dir "$EXPRESSIVE_MODEL_DIR" \
    --output_path "$TEST_OUT/expressive_eng_spa.wav" || return $?
  echo "Output written to $TEST_OUT/expressive_eng_spa.wav"
}

run_test_demo_offline() {
  echo "=== test-demo-offline: launch demo-offline both, probe both ports ==="
  for p in 7860 7862; do
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "(:|\.)$p\$"; then
      echo "  ERROR: port $p already bound — stop the existing demo first."
      return 1
    fi
  done

  local log_dir=$TEST_OUT/demo-logs
  mkdir -p "$log_dir"
  local log=$log_dir/test-demo-offline.log
  : > "$log"

  bash "$SCRIPT_DIR/start.sh" demo-offline both > "$log" 2>&1 &
  local demo_pid=$!
  # RETURN trap fires whether we exit via PASS, FAIL, or set -e — guarantees
  # we don't leak the python apps holding 7860/7862.
  trap "kill_listeners_on 7860 7862; kill $demo_pid 2>/dev/null || true" RETURN

  local timeout=180
  local start=$EPOCHSECONDS
  local m4t_ok=0 expr_ok=0
  while (( EPOCHSECONDS - start < timeout )); do
    if ! kill -0 "$demo_pid" 2>/dev/null && (( m4t_ok == 0 || expr_ok == 0 )); then
      echo "  FAIL: demo process exited before both ports came up. Tail of $log:"
      tail -40 "$log" | sed 's/^/    /'
      return 1
    fi
    if (( m4t_ok == 0 )); then
      if curl -fsS -o /dev/null --max-time 2 http://127.0.0.1:7860/config 2>/dev/null; then
        m4t_ok=1
      fi
    fi
    if (( expr_ok == 0 )); then
      if curl -fsS -o /dev/null --max-time 2 http://127.0.0.1:7862/config 2>/dev/null; then
        expr_ok=1
      fi
    fi
    if (( m4t_ok == 1 && expr_ok == 1 )); then
      local elapsed=$((EPOCHSECONDS - start))
      echo "  OK  M4T         http://127.0.0.1:7860/config  (200)"
      echo "  OK  Expressive  http://127.0.0.1:7862/config  (200)"
      echo "  ready in ${elapsed}s"
      return 0
    fi
    sleep 2
  done

  echo "  FAIL: timed out after ${timeout}s.  m4t_ok=$m4t_ok  expr_ok=$expr_ok"
  echo "  Tail of $log:"
  tail -40 "$log" | sed 's/^/    /'
  return 1
}

run_test_streaming() {
  echo "=== test-streaming: import + asset resolution smoke test ==="
  python - <<'PY'
import sys
import torch
import fairseq2
from fairseq2.assets import asset_store
from seamless_communication.inference import Translator  # noqa: F401
print(f"  torch={torch.__version__}  fairseq2={fairseq2.__version__}")

cards = ["seamless_streaming_unity", "seamless_streaming_monotonic_decoder"]
missing = []
for name in cards:
    try:
        card = asset_store.retrieve_card(name)
        ckpt = card.field("checkpoint").as_uri()
        print(f"  OK  {name}  -> {ckpt}")
    except Exception as e:
        missing.append((name, str(e)))
        print(f"  FAIL {name}: {e}")

try:
    import seamless_communication.streaming.agents  # noqa: F401
    print("  OK  seamless_communication.streaming.agents imports")
except Exception as e:
    missing.append(("streaming.agents import", str(e)))
    print(f"  FAIL streaming.agents import: {e}")

sys.exit(1 if missing else 0)
PY
}

case "$MODE" in
  m4t)
    exec m4t_predict "$@"
    ;;
  expressive)
    check_expressive_weights
    exec expressivity_predict --vocoder_name vocoder_pretssel --gated-model-dir "$EXPRESSIVE_MODEL_DIR" "$@"
    ;;
  try)
    INPUT=${1:-}
    TGT_LANG=${2:-spa}
    if [[ -z "$INPUT" ]]; then
      echo "Usage: $0 try <input.wav> [tgt_lang]"
      exit 1
    fi
    if [[ ! -f "$INPUT" ]]; then
      echo "ERROR: file not found: $INPUT"
      exit 1
    fi
    check_expressive_weights || exit $?
    STEM=$(basename "$INPUT")
    STEM=${STEM%.*}
    M4T_OUT="$TEST_OUT/${STEM}__m4t_s2st_${TGT_LANG}.wav"
    EXPR_OUT="$TEST_OUT/${STEM}__expressive_s2st_${TGT_LANG}.wav"

    echo "=== M4T S2TT: eng -> $TGT_LANG ==="
    m4t_predict "$INPUT" --task S2TT --tgt_lang "$TGT_LANG" \
      --model_name seamlessM4T_v2_large || exit $?

    echo
    echo "=== M4T S2ST: eng -> $TGT_LANG  ($M4T_OUT) ==="
    m4t_predict "$INPUT" --task S2ST --tgt_lang "$TGT_LANG" \
      --model_name seamlessM4T_v2_large --output_path "$M4T_OUT" || exit $?

    echo
    echo "=== SeamlessExpressive S2ST: eng -> $TGT_LANG  ($EXPR_OUT) ==="
    expressivity_predict "$INPUT" --task S2ST --src_lang eng --tgt_lang "$TGT_LANG" \
      --vocoder_name vocoder_pretssel \
      --gated-model-dir "$EXPRESSIVE_MODEL_DIR" \
      --output_path "$EXPR_OUT" || exit $?

    echo
    echo "=== outputs — compare these side-by-side ==="
    echo "  M4T:        $M4T_OUT"
    echo "  Expressive: $EXPR_OUT"
    ;;
  demo-offline)
    CHOICE=${1:-both}
    case "$CHOICE" in m4t|expressive|both) ;;
      *) echo "Usage: $0 demo-offline [m4t|expressive|both]"; exit 1;;
    esac
    ensure_demo_deps || exit $?
    if [[ "$CHOICE" == "expressive" || "$CHOICE" == "both" ]]; then
      check_expressive_weights || exit $?
    fi
    python "$SCRIPT_DIR/demo/setup_checkpoints.py" || exit $?

    LAUNCHER=$SCRIPT_DIR/demo/launch_offline.sh
    LOG_DIR=$TEST_OUT/demo-logs
    mkdir -p "$LOG_DIR"

    echo
    echo "=== demo-offline: $CHOICE ==="
    if [[ "$CHOICE" == "both" ]]; then
      print_proxy_url 7860 "M4T:       "
      print_proxy_url 7862 "Expressive:"
      echo "(both apps share the venv; first request to each pays a model-load cost)"
      echo
      nohup "$LAUNCHER" m4t > "$LOG_DIR/m4t.log" 2>&1 &
      M4T_PID=$!
      echo "[demo-offline] m4t backgrounded as pid $M4T_PID  (log: $LOG_DIR/m4t.log)"
      # NOT `exec` — exec drops the trap, leaking the m4t background process on Ctrl-C.
      trap 'echo "[demo-offline] stopping m4t pid '"$M4T_PID"'"; kill '"$M4T_PID"' 2>/dev/null || true' EXIT INT TERM
      "$LAUNCHER" expressive
    else
      if [[ "$CHOICE" == "expressive" ]]; then
        port=7862; label="Expressive:"
      else
        port=7860; label="M4T:       "
      fi
      print_proxy_url "$port" "$label"
      echo
      exec "$LAUNCHER" "$CHOICE"
    fi
    ;;
  demo-streaming)
    CHOICE=${1:-expressive}
    case "$CHOICE" in expressive|non-expressive) ;;
      *) echo "Usage: $0 demo-streaming [expressive|non-expressive]"; exit 1;;
    esac
    ensure_demo_deps || exit $?
    if [[ "$CHOICE" == "expressive" ]]; then
      check_expressive_weights || exit $?
    fi
    check_streaming_models || exit $?
    print_proxy_url 7860 "Streaming demo:"
    echo "(microphone access requires HTTPS — use the proxy URL, not raw 0.0.0.0)"
    echo
    exec "$SCRIPT_DIR/demo/launch_streaming.sh" "$CHOICE"
    ;;
  claude)
    if ! command -v claude >/dev/null; then
      echo "ERROR: claude CLI not found at $NPM_CONFIG_PREFIX/bin/claude"
      echo "Re-run /workspace/install.sh (Section 1 installs it)."
      exit 1
    fi
    exec claude "$@"
    ;;
  test-m4t)
    run_test_m4t
    ;;
  test-expressive)
    run_test_expressive
    ;;
  test-streaming)
    run_test_streaming
    ;;
  test-demo-offline)
    run_test_demo_offline
    ;;
  test-all)
    declare -A results
    # demo_offline is heavy (~150s — loads both apps into GPU + serves HTTP);
    # ordered last so a fast-fail in the inference smokes surfaces first.
    for t in m4t expressive streaming demo_offline; do
      display=${t//_/-}
      echo
      echo "############################################"
      echo "# test-$display"
      echo "############################################"
      if run_test_$t; then
        results[$t]=PASS
      else
        results[$t]=FAIL
      fi
    done
    echo
    echo "=== test-all summary ==="
    fail=0
    for t in m4t expressive streaming demo_offline; do
      display=${t//_/-}
      printf "  %-13s %s\n" "$display" "${results[$t]}"
      [[ "${results[$t]}" == "FAIL" ]] && fail=1
    done
    exit $fail
    ;;
  shell)
    exec bash
    ;;
  python)
    exec python "$@"
    ;;
  *)
    echo "Usage: $0 {m4t|expressive|try|demo-offline|demo-streaming|claude|test-m4t|test-expressive|test-streaming|test-demo-offline|test-all|shell|python} [args...]"
    exit 1
    ;;
esac
