"""Local text-to-image generation utilities using diffusers."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from diffusers.utils import logging as diffusers_logging

from config import GenerationConfig

# Quieter pipelines by default
logging.getLogger("diffusers").setLevel(logging.ERROR)
diffusers_logging.set_verbosity_error()


@dataclass
class PipelineBundle:
    pipeline: DiffusionPipeline
    face_adapter_loaded: bool = False


def _maybe_enable_xformers(pipe: DiffusionPipeline) -> None:
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        # xformers is optional; ignore if unavailable
        pass


def _apply_memory_saving(pipe: DiffusionPipeline, low_vram: bool) -> None:
    if low_vram:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception:
            # CPU offload requires accelerate with CUDA; safe to skip if unsupported
            pass


def init_pipeline(config: GenerationConfig) -> PipelineBundle:
    pipe = AutoPipelineForText2Image.from_pretrained(
        config.model_id,
        revision=config.revision,
        torch_dtype=config.dtype,
        variant=config.variant,
        use_safetensors=True,
    )

    if not config.safety_checker and hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
        pipe.requires_safety_checker = False

    face_adapter_loaded = False
    if config.face_adapter:
        try:
            subfolder = "models" if config.face_adapter.base == "sdxl" else None
            pipe.load_ip_adapter(
                config.face_adapter.repo_id,
                subfolder=subfolder,
                weight_name=config.face_adapter.weight_name,
            )
            face_adapter_loaded = True
        except Exception as exc:
            logging.warning(f"Face adapter load failed ({exc}); proceeding with text-only generation.")

    _apply_memory_saving(pipe, config.low_vram)
    if not config.low_vram:
        _maybe_enable_xformers(pipe)

    pipe = pipe.to(config.device)
    pipe.set_progress_bar_config(disable=True)
    return PipelineBundle(pipeline=pipe, face_adapter_loaded=face_adapter_loaded)


def generate_image(
    bundle: PipelineBundle,
    prompt: str,
    config: GenerationConfig,
    seed: Optional[int] = None,
    negative_prompt: Optional[str] = None,
    face_image: Optional["Image.Image"] = None,
) -> "Image.Image":
    generator = None
    if seed is not None:
        generator = torch.Generator(device=config.device)
        generator.manual_seed(seed)

    ip_adapter_image_embeds = None
    if face_image is not None and bundle.face_adapter_loaded:
        try:
            ip_adapter_image_embeds = bundle.pipeline.prepare_ip_adapter_image_embeds(
                image=face_image, device=config.device, num_images_per_prompt=1
            )
        except Exception as exc:
            logging.warning(f"Face embedding failed ({exc}); falling back to text-only for this job.")
            ip_adapter_image_embeds = None

    result = bundle.pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt or config.negative_prompt or None,
        height=config.height,
        width=config.width,
        num_inference_steps=config.steps,
        guidance_scale=config.guidance_scale,
        generator=generator,
        ip_adapter_image_embeds=ip_adapter_image_embeds,
    )
    return result.images[0]
