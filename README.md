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
| **Subtotal (streaming,