#!/usr/bin/env bash
# Seamless Translation PoC — runtime entry point.
# Sets up env vars, activates the venv, dispatches to the requested mode.
#
# Usage:
#   ./start.sh m4t <audio.wav> --task {ASR,S2ST,S2TT} --tgt_lang <lang> [...]
#   ./start.sh expressive <audio.wav> --tgt_lang <lang> [...]
#   ./start.sh try <audio.wav> [tgt_lang]    M4T S2TT + M4T S2ST + expressive S2ST, side-by-side
#   ./start.sh streaming                 streaming server (TODO — see test-streaming for smoke test)
#   ./start.sh claude                    Claude Code CLI with volume-backed config
#   ./start.sh test-m4t                  smoke-test M4T v2 against reference WAV
#   ./start.sh test-expressive           smoke-test SeamlessExpressive against reference WAV
#   ./start.sh test-streaming            smoke-test streaming stack (import + asset resolution)
#   ./start.sh test-all                  run all three smoke tests, summarize
#   ./start.sh shell                     bash with env + venv active
#   ./start.sh python [args...]          python with env + venv active

set -euo pipefail

WORKSPACE=/workspace
VENV_DIR=$WORKSPACE/envs/seamless

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
  streaming)
    echo "streaming mode not yet wired up — coming in next phase"
    echo "(for a stack smoke test, try: $0 test-streaming)"
    exit 1
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
  test-all)
    declare -A results
    for t in m4t expressive streaming; do
      echo
      echo "############################################"
      echo "# test-$t"
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
    for t in m4t expressive streaming; do
      printf "  %-12s %s\n" "$t" "${results[$t]}"
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
    echo "Usage: $0 {m4t|expressive|try|streaming|claude|test-m4t|test-expressive|test-streaming|test-all|shell|python} [args...]"
    exit 1
    ;;
esac
