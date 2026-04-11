"""Generate an image with FLUX.2-klein-4B using fouroversix NVFP4 quantization."""

import time

import torch
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from fouroversix.diffusers import FourOverSixConfig

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
PROMPT = (
    "A cozy bookshop in Tokyo at night, warm light spilling onto a rain-slicked "
    "street, cherry blossom petals drifting through the air, Studio Ghibli style"
)
OUTPUT_PATH = "/workspace/fouroversix/flux_klein_output.png"

print(f"Model:  {MODEL_ID}")
print(f"Prompt: {PROMPT}")
print()

quant_config = FourOverSixConfig()
print(f"Quantization config: {quant_config.to_dict()}")

print("Loading transformer with NVFP4 quantization...")
t0 = time.time()
transformer = Flux2Transformer2DModel.from_pretrained(
    MODEL_ID,
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)
print(f"  Transformer loaded in {time.time() - t0:.1f}s")

print("Loading pipeline with quantized transformer...")
t0 = time.time()
pipe = Flux2KleinPipeline.from_pretrained(
    MODEL_ID,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
print(f"  Pipeline loaded in {time.time() - t0:.1f}s")

free, total = torch.cuda.mem_get_info()
print(f"  VRAM: {(total - free) / 1e9:.1f} GB used / {total / 1e9:.1f} GB total")

print("Generating image (4 steps)...")
t0 = time.time()
image = pipe(
    prompt=PROMPT,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]
elapsed = time.time() - t0
print(f"  Generated in {elapsed:.1f}s")

image.save(OUTPUT_PATH)
print(f"  Saved to {OUTPUT_PATH}")
