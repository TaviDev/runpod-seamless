# runpod-seamless

Scripts for running Meta's [seamless_communication](https://github.com/facebookresearch/seamless_communication) (SeamlessM4T v2) on RunPod with a persistent network volume.

## Stack

- RunPod template: `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
- GPU: RTX 4090 (tested), should work on anything Ampere+ with ≥ 12 GB VRAM
- Volume: `seamless-poc`, ≥ 40 GB (80 GB recommended)
- torch 2.1.1 + cu121, fairseq2 0.2.0 (+cu121 variant), numpy < 2
- `seamless_communication` from source, pinned against upstream commit 90e2b57

## Why these exact versions

`seamless_communication/setup.py` pins `fairseq2==0.2.*`. That version of fairseq2n (the native C++ component) strictly requires `torch==2.1.1`. FAIR's wheel index at `https://fair.pkg.atmeta.com/fairseq2/whl/pt2.1.1/cu121/` still hosts the `+cu121` variants. Deviating from these versions produces either resolver failures (safe) or C++ ABI segfaults at runtime (unsafe). Don't.

## Fresh pod from scratch

```
cd /workspace
git clone https://github.com/TaviDev/runpod-seamless.git .scripts
bash .scripts/install.sh
```

Expected time on fresh pod with attached but empty volume: **~5 min**.
Expected time on fresh pod with populated volume (after redeploy): **~1 min** (all sections no-op except apt).

## Run an inference

```
/workspace/start.sh m4t \
  /workspace/seamless_communication/demo/dino_pretssel/audios/employee_eng_spa/ref/clean_spk1_default_00240.wav \
  --task S2TT --tgt_lang spa \
  --model_name seamlessM4T_v2_large
```

First run downloads ~8.5 GB of M4T v2 weights to `/workspace/models/fairseq2-cache/` (~80s on RunPod). Subsequent runs skip the download.

Tasks: `ASR`, `S2TT` (speech-to-text translation), `S2ST` (speech-to-speech).

## Streaming mode

Not yet implemented in `start.sh`. TODO.

## Files on the volume

```
/workspace/
├── .scripts/                          this repo
├── install.sh, start.sh               convenience copies
├── envs/seamless/                     Python venv (~14 GB)
├── seamless_communication/            cloned repo, -e installed (~200 MB)
└── models/
    ├── fairseq2-cache/                lazy-populated by fairseq2 (~9 GB per model family)
    └── hf-cache/                      HF CLI config (token), ~1 MB
```

## Persistence behavior

- **Stop/start:** everything preserved, no reinstall needed. Just `./start.sh ...`.
- **Terminate/redeploy with same volume:** `install.sh` needed again to re-install apt deps and repopulate Python's `-e .` install; venv and models on volume survive. Total time ~1 min.
- **Fresh volume:** full reinstall, ~5 min.