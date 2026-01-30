"""Configuration helpers for local text-to-image generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import torch


@dataclass
class DeviceInfo:
    device: torch.device
    backend: str  # 'cuda', 'rocm', 'cpu', 'mps'
    name: str
    total_vram_gb: Optional[float]


@dataclass
class ModelPreset:
    model_id: str
    revision: Optional[str]
    height: int
    width: int
    steps: int
    guidance_scale: float
    variant: Optional[str]
    default_negative_prompt: str


MODEL_PRESETS = {
    "sdxl": ModelPreset(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        revision=None,
        height=1024,
        width=1024,
        steps=25,
        guidance_scale=7.0,
        variant="fp16",
        default_negative_prompt="",
    ),
    "sdxl-turbo": ModelPreset(
        model_id="stabilityai/sdxl-turbo",
        revision=None,
        height=1024,
        width=1024,
        steps=4,
        guidance_scale=0.0,
        variant="fp16",
        default_negative_prompt="",
    ),
    "sd15": ModelPreset(
        model_id="runwayml/stable-diffusion-v1-5",
        revision=None,
        height=512,
        width=512,
        steps=30,
        guidance_scale=7.5,
        variant=None,
        default_negative_prompt="",
    ),
}

@dataclass
class FaceAdapterSpec:
    key: str
    repo_id: str
    weight_name: str
    base: str  # 'sdxl' or 'sd15'


FACE_ADAPTERS: Dict[str, FaceAdapterSpec] = {
    "faceid-sdxl": FaceAdapterSpec(
        key="faceid-sdxl",
        repo_id="h94/IP-Adapter-FaceID",
        weight_name="ip-adapter-faceid-plusv2_sdxl.bin",
        base="sdxl",
    ),
    "faceid-sd15": FaceAdapterSpec(
        key="faceid-sd15",
        repo_id="h94/IP-Adapter-FaceID",
        weight_name="ip-adapter-faceid-plusv2_sd15.bin",
        base="sd15",
    ),
}


def detect_device(explicit: Optional[str] = None) -> DeviceInfo:
    if explicit:
        device = torch.device(explicit)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    backend = "cpu"
    total_vram = None
    name = "cpu"

    if device.type == "cuda":
        backend = "rocm" if torch.version.hip is not None else "cuda"
        props = torch.cuda.get_device_properties(device)
        total_vram = round(props.total_memory / (1024 ** 3), 2)
        name = props.name
    elif device.type == "mps":
        backend = "mps"
        name = "apple-mps"
    return DeviceInfo(device=device, backend=backend, name=name, total_vram_gb=total_vram)


@dataclass
class GenerationConfig:
    preset_key: str
    model_id: str
    revision: Optional[str]
    device: torch.device
    dtype: torch.dtype
    height: int
    width: int
    steps: int
    guidance_scale: float
    negative_prompt: str
    low_vram: bool
    safety_checker: bool
    seed: Optional[int]
    variant: Optional[str]
    face_adapter: Optional[FaceAdapterSpec]


def choose_preset(info: DeviceInfo, requested: Optional[str], low_vram: bool) -> str:
    if requested in MODEL_PRESETS:
        return requested

    if info.backend in ("cuda", "rocm") and info.total_vram_gb is not None:
        vram = info.total_vram_gb
        if info.backend == "rocm" and vram >= 16:
            return "sdxl"
        if info.backend == "cuda" and vram >= 12:
            return "sdxl"
        if 8 <= vram < 12:
            return "sdxl-turbo"
        return "sd15"

    return "sd15"


def build_generation_config(
    info: DeviceInfo,
    requested_model: Optional[str],
    requested_adapter: Optional[str],
    low_vram: bool,
    steps: Optional[int],
    guidance_scale: Optional[float],
    negative_prompt: str,
    safety_checker: bool,
    seed: Optional[int],
) -> GenerationConfig:
    preset_key = choose_preset(info, requested_model, low_vram)
    preset = MODEL_PRESETS[preset_key]

    # Adapter selection
    face_adapter = None
    if requested_adapter and requested_adapter != "none":
        spec = FACE_ADAPTERS.get(requested_adapter)
        if spec and spec.base in preset_key:
            face_adapter = spec
    elif preset_key.startswith("sdxl"):
        face_adapter = FACE_ADAPTERS.get("faceid-sdxl")
    elif preset_key.startswith("sd"):
        face_adapter = FACE_ADAPTERS.get("faceid-sd15")

    height, width = preset.height, preset.width
    if low_vram:
        height = min(height, 768 if preset_key.startswith("sdxl") else 512)
        width = min(width, 768 if preset_key.startswith("sdxl") else 512)

    dtype = torch.float16 if info.device.type != "cpu" else torch.float32

    return GenerationConfig(
        preset_key=preset_key,
        model_id=preset.model_id,
        revision=preset.revision,
        device=info.device,
        dtype=dtype,
        height=height,
        width=width,
        steps=steps or preset.steps,
        guidance_scale=guidance_scale if guidance_scale is not None else preset.guidance_scale,
        negative_prompt=negative_prompt or preset.default_negative_prompt,
        low_vram=low_vram,
        safety_checker=safety_checker,
        seed=seed,
        variant=preset.variant,
        face_adapter=face_adapter,
    )
