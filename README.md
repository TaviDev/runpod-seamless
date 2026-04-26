# runpod-seamless

Scripts for running Meta's [seamless_communication](https://github.com/facebookresearch/seamless_communication) (SeamlessM4T v2 + streaming) on RunPod with a persistent network volume.

## Stack

- **RunPod template:** `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
- **GPU:** RTX 4090 (tested), should work on anything Ampere+ with ≥ 12 GB VRAM
- **Python:** 3.10, **torch:** 2.1.1+cu121, **fairseq2:** 0.2.0+cu121, **numpy:** < 2
- `seamless_communication` installed from source, pinned against upstream commit `90e2b57`

## Why these exact versions

`seamless_communication/setup.py` pins `fairseq2==0.2.*`. That version of `fairseq2n` (the native C++ component) strictly requires `torch==2.1.1`. FAIR's wheel index at `https://fair.pkg.atmeta.com/fairseq2/whl/pt2.1.1/cu121/` still hosts the `+cu121` variants.

Deviating from these versions produces either resolver failures (safe, loud) or C++ ABI segfaults at runtime (unsafe, silent). Don't.

## Sizing

### Container disk

**Minimum 40 GB.** RunPod's default 20 GB is too small — pip's download cache alone can hit 3–5 GB during install, plus apt, plus container overhead.

The container disk holds only transient things (download caches, apt, `/tmp`). It's wiped on pod terminate, but survives stop/start. Nothing important lives there.

### Network volume

Everything that needs to persist lives here (venv, repo, model cache). Sizing depends on how many languages you need speech encoders for — **each additional language adds ~8.6 GB** (SONAR speech encoder per language).

| Component | Size | Notes |
|---|---|---|
| venv | 14 GB | fixed |
| seamless_communication repo | 0.2 GB | fixed |
| M4T v2 + vocoder + tokenizers | 8.7 GB | **supports all 100+ languages, no per-lang files** |
| Streaming unity + monotonic decoder | 8 GB | fixed, for streaming mode |
| SONAR text encoder + 2 decoders | 11 GB | fixed, shared across languages |
| SONAR speech encoders | 8.6 GB × N | one per source language for streaming |
| **Subtotal (no streaming)** | **~23 GB** | M4T-only, any target language |
| **Subtotal (streaming, 1 source lang)** | **~50 GB** | M4T + streaming + 1 SONAR encoder |
| **Subtotal (streaming, all langs)** | **~110 GB** | M4T + streaming + 7 SONAR encoders |

## Cold pod bringup

Pre-reqs:
- `HF_TOKEN` set in the pod template env (read access; accept the model
  terms at <https://huggingface.co/facebook/seamless-expressive> first).
- Container disk ≥ 40 GB; network volume mounted at `/workspace`.
- Pod template: `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`.

**Direct path** (run install.sh in your terminal, watch the logs yourself):

```bash
cd /workspace
git clone https://github.com/TaviDev/runpod-seamless.git .scripts
bash .scripts/install.sh         # ~25–35 min cold; ~22 GB gated weights + venv + React build
/workspace/start.sh test-all     # smoke: m4t + expressive + streaming + demo-offline
```

**With claude-code driving** (claude runs install.sh and reports back):

```bash
cd /workspace
git clone https://github.com/TaviDev/runpod-seamless.git .scripts
bash .scripts/bootstrap-claude.sh   # installs node 20 + claude-code, then exec's claude
# in claude:  "run bash /workspace/.scripts/install.sh, then /workspace/start.sh test-all"
```

`bootstrap-claude.sh` is idempotent — re-runs short-circuit on each step
that's already done. Both `node` and the `claude` CLI live under the
volume-backed `/workspace/npm-global/`, so they survive pod stop/start.

`install.sh` is idempotent: every section has a guard, so re-runs after a
partial install pick up where they left off. `HF_TOKEN` is read from the env
on first install; thereafter `hf auth login` writes it to
`/workspace/models/hf-cache/token`, which persists on the volume across pod
stop/start. Subsequent installs/redeploys don't need the env var.

Add `7860` and `7862` as **HTTP Ports** in the pod UI (Edit Pod → HTTP
Ports) so the demos are reachable through the RunPod HTTPS proxy at
`https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net/`.

Use **HTTP Ports**, not TCP Ports. The HTTP proxy terminates TLS, returns a
`https://…proxy.runpod.net/` URL (required for browser mic access), and
forwards WebSocket upgrades — so the streaming demo's Socket.IO traffic
goes through the same HTTP Port without a separate TCP entry. TCP Ports
are raw forwarding (e.g. SSH on `22`); they're not what the demos need.

Demo commands (after `test-all` passes):

```bash
/workspace/start.sh demo-offline both        # M4T :7860 + Expressive :7862, side by side
/workspace/start.sh demo-streaming           # streaming on :7860 — stop demo-offline m4t first
```

## Demos

Two browser-accessible demos, both via RunPod's HTTPS proxy. Add the listed
ports as **HTTP Ports** on the pod (Edit Pod → HTTP Ports) — each port becomes
reachable at `https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net/`.

### Offline Gradio (`demo-offline`)

Two unmodified upstream Gradio apps, side-by-side:

| Mode | Port | Notes |
|---|---|---|
| `m4t` | 7860 | M4T v2 — S2ST/S2TT/T2ST/T2TT/ASR, 105 langs |
| `expressive` | 7862 | SeamlessExpressive — S2ST only, 4 langs (eng/fra/deu/spa), preserves prosody |

```
./start.sh demo-offline both          # default
./start.sh demo-offline m4t           # M4T only
./start.sh demo-offline expressive    # expressive only
```

Port `7861` is reserved by the pod template's nginx, hence `7862` for
expressive. First request to each app pays a one-time model-load cost
(~30–60 s for `Translator()`).

### Streaming (`demo-streaming`)

Meta's HF Space at `facebook/seamless-streaming` — Starlette + Socket.IO
backend serving a Vite/React frontend on port `7860`. Real-time mic input.

```
./start.sh demo-streaming             # expressive (default; USE_EXPRESSIVE_MODEL=1)
./start.sh demo-streaming non-expressive
```

Notes:
- **Microphone access requires HTTPS**, so use the proxy URL — not raw
  `http://0.0.0.0:7860`.
- The proxy is Cloudflare-fronted with a **100-second idle timeout**; long
  silences may drop the WebSocket. Browsers usually reconnect automatically.
- WebSocket / Socket.IO traffic is forwarded by the HTTP proxy — no
  separate TCP Port needed; the same `7860` HTTP Port handles both the
  Vite/React asset serving and the Socket.IO upgrade.
- `demo-streaming` and `demo-offline m4t` both use port 7860; run one at a
  time.
- Background-noise denoising is not wired up yet — speak into a quiet mic for
  best results.

### Smoke tests

`test-all` covers all four layers; the demo-offline smoke runs last because
it's heavy (~150 s cold). Standalone modes are useful when iterating on a
single layer:

```
/workspace/start.sh test-m4t              # M4T v2 inference (S2TT eng→spa)
/workspace/start.sh test-expressive       # SeamlessExpressive inference (S2ST eng→spa)
/workspace/start.sh test-streaming        # streaming imports + asset URLs (no inference)
/workspace/start.sh test-demo-offline     # launch demo-offline both, probe :7860 + :7862
/workspace/start.sh test-all              # all four, summarized
```

`test-demo-offline` polls `/config` on each Gradio app with a 180 s
timeout and tears the apps down via `kill_listeners_on` regardless of
pass/fail, so it's safe to run while another demo isn't already on those
ports.

### First-time setup

`install.sh` handles everything both demos need — Section 4 installs gradio
+ a pinned HTTP stack (`starlette<0.40`, `fastapi<0.110`, `uvicorn<0.35`),
and Section 5 clones the streaming HF Space and builds the React frontend.

The HTTP-stack pins are required: pip's resolver otherwise pulls
`starlette` ≥ 1.0, which breaks gradio 4.5's template render with
`TypeError: unhashable type: 'dict'`. `start.sh`'s `ensure_demo_deps`
helper is a runtime fallback for volumes that predate this install path —
on a fresh `install.sh` run, it's a no-op.
