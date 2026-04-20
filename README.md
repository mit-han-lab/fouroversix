# Four Over Six (4/6)

[![arXiv](https://img.shields.io/badge/arXiv-2512.02010-b31b1b.svg)](https://arxiv.org/abs/2512.02010)

_Improving the accuracy of block-scaled quantization with Adaptive Block Scaling._

![](/assets/four-over-six.png)

This repository contains kernels for efficient block-scaled quantization and matrix multiplication, and scripts for running post-training quantization experiments with our methods, 4/6 and IF data types.
If you have any questions, please get in touch or submit an issue.

## Setup

**Requirements:**

- Python version 3.10 or newer
- CUDA toolkit 12.8 or newer
- PyTorch version 2.8 or newer

**Install dependencies:**

```bash
pip install ninja packaging psutil "setuptools>=77.0.3"
```

**Install fouroversix:**

```bash
pip install fouroversix --no-build-isolation
```

Alternatively, you can compile from source:

```bash
pip install --no-build-isolation -e .
```

To speed up build times, set `CUDA_ARCHS=100` to only compile kernels for B-series GPUs (i.e. B200, GB200, GB300), or `CUDA_ARCHS=120` for RTX 50 and 60 Series GPUs (i.e. RTX 5090, RTX 6000).

Also, if you don't have a Blackwell GPU, you may use our reference implementation, which is slow but helpful for testing, by setting `SKIP_CUDA_BUILD=1` before running `pip install`.

### PTQ Experiments (New)

In our newest work, Adaptive Block-Scaled Data Types, our experiments were run with a new vLLM-based PTQ setup.
This setup currently does not support combining 4/6 or IF data types with other PTQ methods such as AWQ, GPTQ, or SmoothQuant.
To run experiments with round-to-nearest quantization and vLLM, follow these instructions.
Otherwise, scroll down to set up PTQ experiments with our older Hugging Face-based codebase.

First, install necessary dependencies:

```bash
pip install vllm
pip install "transformers @ git+https://github.com/jackcook/transformers.git@jack/dtypes"
```

You will also need to install [Flash Attention](https://github.com/dao-ailab/flash-attention).
Then, launch an experiment with the following sample command:

```bash
python -m scripts.ptq --model Qwen/Qwen3.5-35B-A3B --quantization-scheme if4 --tasks arc_easy
```

You may select quantization schemes and tasks from the enums in scripts/ptq/utils.py.

### PTQ Experiments (Old)

To run PTQ experiments using our old setup, make sure to install our test dependencies using either:

```bash
pip install "fouroversix[evals]" --no-build-isolation

# Or, if installing from source:
pip install --no-build-isolation -e ".[evals]"
```

Also, make sure all submodules are pulled and up to date:

```bash
git submodule update --init
```

Then, install dependencies for each PTQ method as needed, following the instructions [here](/docs/ptq.md), before running one of the following commands:

```bash
# Round-to-nearest quantization with 4/6:
python -m scripts.ptq_hf --model-name meta-llama/Llama-3.2-1B --ptq-method rtn --task wikitext

# Standard NVFP4 round-to-nearest (RTN) quantization:
python -m scripts.ptq_hf --model-name meta-llama/Llama-3.2-1B --ptq-method rtn --task wikitext --a-scale-rule static_6 --w-scale-rule static_6

# AWQ with 4/6:
python -m scripts.ptq_hf --model-name meta-llama/Llama-3.2-1B --ptq-method awq --task wikitext

# High-precision baseline, no NVFP4 quantization:
python -m scripts.ptq_hf --model-name meta-llama/Llama-3.2-1B --ptq-method high_precision --task wikitext
```

If you would prefer not to worry about setting up your local environment, or about acquiring a Blackwell GPU to run your experiments faster, you may run PTQ experiments on [Modal](https://modal.com/) by adding the `--modal` flag, and optionally the `--detach` flag which will enable you to CTRL+C.
The first time you launch experiments on Modal, it may take several minutes to build everything, but following commands will reuse the cached images.

## API

### Quantize a Model to NVFP4

```python
from fouroversix import ModelQuantizationConfig, quantize_model
from transformers import AutoModelForCausalLM

# NVFP4 using 4/6 with MSE block selection
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
quantize_model(model)

# Standard NVFP4 round-to-nearest quantization
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
config = ModelQuantizationConfig(scale_rule="static_6")
quantize_model(model, config)
```

### Quantize a Tensor to NVFP4

Check the `quantize` [arguments](https://github.com/mit-han-lab/fouroversix/blob/f1b78701c753ea49c091ac39d85c5753b703f5ca/src/fouroversix/frontend.py#L72) for more details about how you can enable certain features during quantization, such as stochastic rounding or 2D block quantization.

```python
import torch
from fouroversix import QuantizationConfig, quantize

x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
x_quantized = quantize(x)

# Standard NVFP4 round-to-nearest quantization
config = QuantizationConfig(scale_rule="static_6")
x_quantized = quantize(x, config)
```

### Quantize a Diffusers Model to NVFP4

Install diffusers support with:

```bash
pip install "fouroversix[diffusers]" --no-build-isolation
```

You can quantize any diffusers model using `quantize_model`:

```python
import torch
from diffusers import FluxTransformer2DModel
from fouroversix import ModelQuantizationConfig, quantize_model

model = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
config = ModelQuantizationConfig(modules_to_not_convert=[])
quantize_model(model, config)
```

Or use the `from_pretrained` integration which automatically quantizes during loading:

```python
import torch
from diffusers import FluxTransformer2DModel
from fouroversix.diffusers import FourOverSixConfig

model = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    quantization_config=FourOverSixConfig(),
)
```

### Multiply Two NVFP4 Tensors

```python
from fouroversix import quantized_matmul

# a and b can be either high-precision BF16 tensors, in which case they will be
# quantized, or low-precision QuantizedTensors if you've already quantized them
# yourself.
out = quantized_matmul(a, b)
```

## Notes

This repository contains three implementations of NVFP4 quantization, each of which has various limitations:

- [CUDA](/src/fouroversix/csrc): Supports most but not all operations needed for efficient NVFP4 training. More operations will be added soon. Requires a Blackwell GPU.
- [Triton](/src/fouroversix/quantize/triton_kernel.py): Supports all operations needed for efficient NVFP4 training, including stochastic rounding, the random Hadamard transform, transposed inputs, and 2D block scaling. Requires a Blackwell GPU.
- [PyTorch](/src/fouroversix/quantize/reference.py): A reference implementation written in PyTorch that can run on any GPU. May have some educational value. Should not be used in real-world use cases.

Our `quantize` function will automatically select one of these backends based on your GPU and the quantization parameters you select.
If you would like to force selection of a specific backend, you may specify it by setting `backend=QuantizeBackend.cuda` in the quantization config passed to `quantize`, or `quantize_backend=QuantizeBackend.cuda` in the layer and model configs passed to `quantize_model`.

## Contributing

We welcome contributions to our repository, but get in touch before making any substantial changes.
Also, please make sure any code changes are compliant with our linter:

```bash
ruff check
```

## Citation

Please use the following BibTeX entries to cite this work:

```bibtex
@misc{cook2025sixaccuratenvfp4quantization,
      title={Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling},
      author={Jack Cook and Junxian Guo and Guangxuan Xiao and Yujun Lin and Song Han},
      year={2025},
      eprint={2512.02010},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.02010},
}

@misc{cook2026adaptiveblockscaleddatatypes,
      title={Adaptive Block-Scaled Data Types}, 
      author={Jack Cook and Hyemin S. Lee and Kathryn Le and Junxian Guo and Giovanni Traverso and Anantha P. Chandrakasan and Song Han},
      year={2026},
      eprint={2603.28765},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.28765}, 
}
```

## License

This repository is available under the MIT license.
See the [LICENSE.md](/LICENSE.md) file for details.
