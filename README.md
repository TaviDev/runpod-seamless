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
- `demo-streaming` and `demo-offline m4t` both use port 7860; run one at a
  time.
- Background-noise denoising is not wired up yet — speak into a quiet mic for
  best results.

### First-time setup

`demo-offline` and `demo-streaming` both call `ensure_demo_deps` on first run,
which installs `gradio~=4.5.0` plus an HTTP stack pinned to the gradio
4.5.0-era window (`starlette<0.40`, `fastapi<0.110`, `uvicorn<0.35`). The
gradio pin matches upstream's demo `requirements.txt`; the HTTP-stack pins
are needed because pip's resolver otherwise pulls in `starlette` ≥ 1.0,
which breaks gradio's template render with `TypeError: unhashable type:
'dict'`.

Streaming additionally needs the cloned Space repo at
`/workspace/seamless-streaming-demo/` and a built React `dist/`. One-time:

```
git clone https://huggingface.co/spaces/facebook/seamless-streaming \
  /workspace/seamless-streaming-demo
cd /workspace/seamless-streaming-demo/streaming-react-app
yarn install && yarn build
```
