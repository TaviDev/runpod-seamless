#!/usr/bin/env python
"""
Build a consolidated CHECKPOINTS_PATH dir for Meta's two demo apps.

Both demo/m4tv2/app.py and demo/expressive/app.py register assets via
InProcAssetMetadataProvider with `file://{CHECKPOINTS_PATH}/<basename>`
URIs. The weights they need live in two different volume locations:

  - fairseq2-cache (downloaded by `m4t_predict` / `expressivity_predict`):
      seamlessM4T_v2_large.pt, vocoder_v2.pt, spm_char_lang38_tc.model
  - hf-gated/seamless-expressive (downloaded via `hf download`):
      m2m_expressive_unity.pt, pretssel_melhifigan_wm-final.pt

This script symlinks all of them into /workspace/models/checkpoints-demo/.
Idempotent — re-running is a no-op if everything is already linked.

fairseq2 paths are resolved at runtime via the asset store, never
hard-coded — the cache dirs are content-addressed and could drift.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CHECKPOINTS_DIR = Path("/workspace/models/checkpoints-demo")
EXPRESSIVE_DIR = Path(os.environ.get(
    "EXPRESSIVE_MODEL_DIR",
    "/workspace/models/hf-gated/seamless-expressive",
))


def link(target: Path, link_path: Path) -> str:
    """Create or refresh a symlink. Returns 'ok' / 'created' / 'updated'."""
    if not target.exists():
        raise FileNotFoundError(f"source missing: {target}")
    if link_path.is_symlink():
        if link_path.resolve() == target.resolve():
            return "ok"
        link_path.unlink()
        link_path.symlink_to(target)
        return "updated"
    if link_path.exists():
        raise FileExistsError(
            f"{link_path} exists and is not a symlink — refusing to overwrite"
        )
    link_path.symlink_to(target)
    return "created"


def resolve_fairseq2_assets() -> dict[str, Path]:
    """Force-download (no-op if cached) the fairseq2 assets the demos need."""
    # Importing seamless_communication registers its asset cards with fairseq2.
    import seamless_communication  # noqa: F401
    from fairseq2.assets import asset_store, download_manager

    out: dict[str, Path] = {}

    m4t_card = asset_store.retrieve_card("seamlessM4T_v2_large")
    out["seamlessM4T_v2_large.pt"] = Path(download_manager.download_checkpoint(
        m4t_card.field("checkpoint").as_uri(), m4t_card.name,
    ))
    out["spm_char_lang38_tc.model"] = Path(download_manager.download_tokenizer(
        m4t_card.field("char_tokenizer").as_uri(), m4t_card.name,
        tokenizer_name="char_tokenizer",
    ))

    voc_card = asset_store.retrieve_card("vocoder_v2")
    out["vocoder_v2.pt"] = Path(download_manager.download_checkpoint(
        voc_card.field("checkpoint").as_uri(), voc_card.name,
    ))

    return out


def main() -> int:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup_checkpoints] target dir: {CHECKPOINTS_DIR}")

    sources: dict[str, Path] = {}

    print("[setup_checkpoints] resolving fairseq2 assets (downloads if needed)...")
    try:
        sources.update(resolve_fairseq2_assets())
    except Exception as e:
        print(f"ERROR: failed to resolve fairseq2 assets: {e}", file=sys.stderr)
        return 2

    if not EXPRESSIVE_DIR.is_dir():
        print(
            f"ERROR: expressive weights dir missing: {EXPRESSIVE_DIR}\n"
            f"  Run: hf download facebook/seamless-expressive --repo-type model \\\n"
            f"         --local-dir {EXPRESSIVE_DIR}",
            file=sys.stderr,
        )
        return 3

    sources["m2m_expressive_unity.pt"] = EXPRESSIVE_DIR / "m2m_expressive_unity.pt"
    sources["pretssel_melhifigan_wm-final.pt"] = (
        EXPRESSIVE_DIR / "pretssel_melhifigan_wm-final.pt"
    )

    failures: list[str] = []
    for basename, src in sources.items():
        dst = CHECKPOINTS_DIR / basename
        try:
            status = link(src, dst)
            print(f"  {status:8s} {basename}  ->  {src}")
        except Exception as e:
            failures.append(f"{basename}: {e}")
            print(f"  FAIL    {basename}: {e}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} symlink(s) failed.", file=sys.stderr)
        return 1

    print(f"\n[setup_checkpoints] OK — {len(sources)} weights linked into {CHECKPOINTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
