"""Benchmark FLUX.2-klein-4B: bf16 baseline vs fouroversix NVFP4 quantization."""

import gc
import time

import torch
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from fouroversix.diffusers import FourOverSixConfig

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
PROMPT = (
    "A cozy bookshop in Tokyo at night, warm light spilling onto a rain-slicked "
    "street, cherry blossom petals drifting through the air, Studio Ghibli style"
)
HEIGHT, WIDTH = 1024, 1024
STEPS = 4
WARMUP_RUNS = 1
BENCH_RUNS = 3
SEED = 42


def flush() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def vram_stats() -> tuple[float, float]:
    allocated = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9
    return allocated, reserved


def generate(pipe: Flux2KleinPipeline, seed: int) -> float:
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    pipe(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        guidance_scale=1.0,
        num_inference_steps=STEPS,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def benchmark(label: str, pipe: Flux2KleinPipeline) -> dict[str, float]:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    for i in range(WARMUP_RUNS):
        print(f"  Warmup {i + 1}/{WARMUP_RUNS}...")
        generate(pipe, SEED + 1000 + i)

    times = []
    for i in range(BENCH_RUNS):
        elapsed = generate(pipe, SEED + i)
        times.append(elapsed)
        print(f"  Run {i + 1}/{BENCH_RUNS}: {elapsed:.3f}s")

    alloc, reserved = vram_stats()
    avg = sum(times) / len(times)
    best = min(times)
    print(f"  Avg: {avg:.3f}s  Best: {best:.3f}s")
    print(f"  Peak VRAM allocated: {alloc:.2f} GB  reserved: {reserved:.2f} GB")
    return {
        "avg": avg,
        "best": best,
        "peak_alloc_gb": alloc,
        "peak_reserved_gb": reserved,
    }


# ── BF16 baseline ──────────────────────────────────────────
print("Loading bf16 baseline pipeline...")
flush()
t0 = time.perf_counter()
pipe_bf16 = Flux2KleinPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)
pipe_bf16.to("cuda")
load_bf16 = time.perf_counter() - t0
print(f"  Loaded in {load_bf16:.1f}s")

results_bf16 = benchmark("BF16 (baseline)", pipe_bf16)
results_bf16["load_time"] = load_bf16

del pipe_bf16
flush()

# ── NVFP4 quantized ───────────────────────────────────────
print("\nLoading NVFP4 quantized pipeline...")
t0 = time.perf_counter()
transformer = Flux2Transformer2DModel.from_pretrained(
    MODEL_ID,
    subfolder="transformer",
    quantization_config=FourOverSixConfig(),
    torch_dtype=torch.bfloat16,
)
pipe_fp4 = Flux2KleinPipeline.from_pretrained(
    MODEL_ID,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe_fp4.to("cuda")
load_fp4 = time.perf_counter() - t0
print(f"  Loaded in {load_fp4:.1f}s")

results_fp4 = benchmark("NVFP4 (fouroversix)", pipe_fp4)
results_fp4["load_time"] = load_fp4

del pipe_fp4
flush()

# ── Summary ────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  COMPARISON SUMMARY")
print(f"{'='*60}")
print(f"{'Metric':<28} {'BF16':>10} {'NVFP4':>10} {'Speedup':>10}")
print(f"{'-'*60}")
print(
    f"{'Load time (s)':<28} {results_bf16['load_time']:>10.2f} "
    f"{results_fp4['load_time']:>10.2f}",
)
print(
    f"{'Avg generation (s)':<28} {results_bf16['avg']:>10.3f} "
    f"{results_fp4['avg']:>10.3f} "
    f"{results_bf16['avg']/results_fp4['avg']:>9.2f}x",
)
print(
    f"{'Best generation (s)':<28} {results_bf16['best']:>10.3f} "
    f"{results_fp4['best']:>10.3f} "
    f"{results_bf16['best']/results_fp4['best']:>9.2f}x",
)
print(
    f"{'Peak VRAM alloc (GB)':<28} {results_bf16['peak_alloc_gb']:>10.2f} "
    f"{results_fp4['peak_alloc_gb']:>10.2f} "
    f"{results_bf16['peak_alloc_gb']/results_fp4['peak_alloc_gb']:>9.2f}x",
)
print(
    f"{'Peak VRAM reserved (GB)':<28} {results_bf16['peak_reserved_gb']:>10.2f} "
    "{results_fp4['peak_reserved_gb']:>10.2f} "
    f"{results_bf16['peak_reserved_gb']/results_fp4['peak_reserved_gb']:>9.2f}x",
)
