"""Test fouroversix diffusers integration with a real model."""

import gc
import time

import torch
from diffusers import PixArtTransformer2DModel
from fouroversix import ModelQuantizationConfig, quantize_model
from fouroversix.diffusers import FourOverSixConfig

MODEL_ID = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
SUBFOLDER = "transformer"
BATCH = 1
SEQ_LEN = 120
HEIGHT, WIDTH = 512, 512
LATENT_CHANNELS = 4
LATENT_H, LATENT_W = HEIGHT // 8, WIDTH // 8


def make_inputs(
    model: PixArtTransformer2DModel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create dummy inputs matching the model's expected shapes."""
    caption_channels = model.config.caption_channels
    latents = torch.randn(
        BATCH, LATENT_CHANNELS, LATENT_H, LATENT_W,
        dtype=torch.bfloat16, device="cuda",
    )
    encoder_hidden_states = torch.randn(
        BATCH, SEQ_LEN, caption_channels,
        dtype=torch.bfloat16, device="cuda",
    )
    timestep = torch.tensor([500.0], device="cuda")
    return latents, encoder_hidden_states, timestep


def run_forward(model: PixArtTransformer2DModel) -> None:
    """Run a forward pass and print results."""
    latents, encoder_hidden_states, timestep = make_inputs(model)
    with torch.no_grad():
        t0 = time.time()
        output = model(
            latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
        )
        torch.cuda.synchronize()
        elapsed = time.time() - t0
    lo = output.sample.min().item()
    hi = output.sample.max().item()
    print(f"  Forward pass: {elapsed:.3f}s")
    print(f"  Output shape: {output.sample.shape}")
    print(f"  Output dtype: {output.sample.dtype}")
    print(f"  Output range: [{lo:.4f}, {hi:.4f}]")


# ── Test 1: quantize_model() path ──────────────────────────
print("=" * 60)
print("Test 1: quantize_model() with keep_master_weights=True")
print("=" * 60)

print(f"Loading {MODEL_ID} in bf16...")
t0 = time.time()
model = PixArtTransformer2DModel.from_pretrained(
    MODEL_ID, subfolder=SUBFOLDER, torch_dtype=torch.bfloat16,
)
model = model.to("cuda")
print(f"  Loaded in {time.time() - t0:.1f}s")

param_count = sum(p.numel() for p in model.parameters())
param_mb = sum(
    p.numel() * p.element_size() for p in model.parameters()
) / 1e6
print(f"  Parameters: {param_count / 1e6:.1f}M ({param_mb:.0f} MB)")

print("Quantizing with quantize_model()...")
t0 = time.time()
config = ModelQuantizationConfig(
    keep_master_weights=True,
    modules_to_not_convert=[],
)
quantize_model(model, config)
print(f"  Quantized in {time.time() - t0:.1f}s")

run_forward(model)

del model
gc.collect()
torch.cuda.empty_cache()


# ── Test 2: from_pretrained + FourOverSixConfig ────────────
print()
print("=" * 60)
print("Test 2: from_pretrained() + FourOverSixConfig")
print("=" * 60)

quant_config = FourOverSixConfig()
print(f"  Config: {quant_config.to_dict()}")

print(f"Loading + quantizing {MODEL_ID} via from_pretrained...")
t0 = time.time()
model = PixArtTransformer2DModel.from_pretrained(
    MODEL_ID, subfolder=SUBFOLDER,
    torch_dtype=torch.bfloat16,
    quantization_config=quant_config,
)
print(f"  Loaded + quantized in {time.time() - t0:.1f}s")
print(
    f"  model.is_quantized = "
    f"{getattr(model, 'is_quantized', 'N/A')}",
)

run_forward(model)

print()
print("All tests passed!")
