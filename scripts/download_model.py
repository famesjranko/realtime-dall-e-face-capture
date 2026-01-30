"""One-time model download helper to satisfy offline requirement."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huggingface_hub import snapshot_download

from config import FACE_ADAPTERS, MODEL_PRESETS, choose_preset, detect_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and cache a diffusion model")
    parser.add_argument("--model", choices=MODEL_PRESETS.keys(), default=None, help="Model preset to cache")
    parser.add_argument("--revision", default=None, help="Optional model revision to pin")
    parser.add_argument("--cache-dir", default=None, help="Cache directory (defaults to HF cache)")
    parser.add_argument("--low-vram", action="store_true", help="Prefer smaller model if auto-selecting")
    parser.add_argument("--face-adapter", choices=["faceid-sdxl", "faceid-sd15"], default=None, help="Download FaceID adapter")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_info = detect_device()
    preset_key = choose_preset(device_info, args.model, args.low_vram)
    preset = MODEL_PRESETS[preset_key]

    print(f"Downloading preset '{preset_key}' -> {preset.model_id}")
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    snapshot_download(
        repo_id=preset.model_id,
        revision=args.revision or preset.revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=False,
        resume_download=True,
    )
    adapter_key = args.face_adapter or ("faceid-sdxl" if preset_key.startswith("sdxl") else "faceid-sd15")
    adapter = FACE_ADAPTERS.get(adapter_key)
    if adapter:
        print(f"Downloading FaceID adapter '{adapter.key}' -> {adapter.repo_id}")
        snapshot_download(
            repo_id=adapter.repo_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=False,
            resume_download=True,
            allow_patterns=[adapter.weight_name, "model_index.json", "*.bin", "*.safetensors"],
        )
    print("Download complete. Subsequent runs can be fully offline.")


if __name__ == "__main__":
    main()
