# Four Over Six (4/6)

[![arXiv](https://img.shields.io/badge/arXiv-2512.02010-b31b1b.svg)](https://arxiv.org/abs/2512.02010)

_A method for improving the accuracy of NVFP4 quantization with Adaptive Block Scaling._

![](/assets/four-over-six.png)

This repository contains kernels for efficient NVFP4 quantization and matrix multiplication, and fast post-training quantization with our method, 4/6.
If you have any questions, please get in touch or submit an issue.

## Setup

To speed up build times, set `CUDA_ARCHS=100` to only compile kernels for B-series GPUs (i.e. B200, GB200, GB300), or `CUDA_ARCHS=120` for RTX 50 and 60 Series GPUs (i.e. RTX 5090, RTX 6000).

```bash
git clone --recursive https://github.com/mit-han-lab/fouroversix.git
cd fouroversix
pip install --no-build-isolation -e ".[tests]"
```

If you don't have a Blackwell GPU, you may use our reference implementation, which is slow but helpful for testing, by setting `SKIP_CUDA_BUILD=1` before running `pip install`.

## API

### Quantize a Model to NVFP4

```python
from fouroversix import AdaptiveBlockScalingRule, apply_ptq
from transformers import AutoModelForCausalLM

# NVFP4 using 4/6 with MSE block selection
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
apply_ptq(model)

# Standard NVFP4 round-to-nearest quantization
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
apply_ptq(
    model,
    a_scale_rule=AdaptiveBlockScalingRule.always_6,
    w_scale_rule=AdaptiveBlockScalingRule.always_6,
)
```

### Quantize a Tensor to NVFP4

Check the `quantize_to_fp4` [arguments](https://github.com/mit-han-lab/fouroversix/blob/f1b78701c753ea49c091ac39d85c5753b703f5ca/src/fouroversix/frontend.py#L72) for more details about how you can enable certain features during quantization, such as stochastic rounding or 2D block quantization.

```python
import torch
from fouroversix import AdaptiveBlockScalingRule, quantize_to_fp4

x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
x_e2m1, x_e4m3, x_normconst = quantize_to_fp4(x)

# Standard NVFP4 round-to-nearest quantization
x_e2m1, x_e4m3, x_normconst = quantize_to_fp4(
    x,
    scale_rule=AdaptiveBlockScalingRule.always_6,
)
```

### Multiply Two NVFP4 Tensors

```python
from fouroversix import fp4_matmul

# Starting from two BF16 tensors with shape (M, K) and (N, K):
out = fp4_matmul(a, b)

# If you've already quantized two tensors A and B as shown above:
out = fp4_matmul(
    a_e2m1=a_e2m1,
    a_sf=a_e4m3,
    a_normconst=a_normconst,
    b_e2m1=b_e2m1,
    b_sf=b_e4m3,
    b_normconst=b_normconst,
)
```

## PTQ Evaluation with LM Evaluation Harness

```bash
# Round-to-nearest quantization with 4/6:
python -m scripts.ptq --model-name meta-llama/Llama-3.2-1B --ptq-method rtn --task wikitext

# Standard NVFP4 round-to-nearest (RTN) quantization:
python -m scripts.ptq --model-name meta-llama/Llama-3.2-1B --ptq-method rtn --task wikitext --a-scale-rule always_6 --w-scale-rule always_6

# AWQ with 4/6:
python -m scripts.ptq --model-name meta-llama/Llama-3.2-1B --ptq-method awq --task wikitext

# High-precision baseline, no NVFP4 quantization:
python -m scripts.ptq --model-name meta-llama/Llama-3.2-1B --ptq-method high_precision --task wikitext
```

If you would prefer not to worry about setting up your local environment, or about acquiring a Blackwell GPU to run your experiments faster, you may run PTQ experiments on [Modal](https://modal.com/) by adding the `--modal` flag, and optionally the `--detach` flag which will enable you to CTRL+C.
The first time you launch experiments on Modal, it may take several minutes to build everything, but following commands will reuse the cached images.

## Notes

This repository contains three implementations of NVFP4 quantization, each of which has various limitations:

- [CUDA](/src/fouroversix/csrc): Only supports forward passes, making it usable for post-training quantization as shown above. Training kernels will be released soon. Requires a Blackwell GPU.
- [Triton](/src/fouroversix/quantize/triton_kernel.py): Slower, but supports all operations needed for efficient NVFP4 training, including stochastic rounding, the random Hadamard transform, transposed inputs, and 2D block scaling. Also requires a Blackwell GPU.
- [PyTorch](/src/fouroversix/quantize/reference.py): A reference implementation written in PyTorch that can run on any GPU. May have some educational value. Should not be used in real-world use cases.

These three implementations have very subtle numerical differences, which we are working on fixing.
Our `quantize_to_fp4` function will automatically select one of these backends based on your GPU and the quantization parameters you select.
If you would like to force selection of a specific backend, you may specify it by setting `backend=QuantizeBackend.cuda` in `quantize_to_fp4`, or `a_quantize_kwargs={"backend": QuantizeBackend.cuda}, w_quantize_kwargs={"backend": QuantizeBackend.cuda}` in `apply_ptq`.

### TODOs

In the coming days and weeks, we will be updating our implementation and publishing more code.
Here are our highest-priority items at the moment:

- [ ] Match numerics of PyTorch and Triton backends to the CUDA backend
- [ ] Add support for other options (MXFP4, stochastic rounding, RHT, 2D block scaling, transposed inputs) in the CUDA implementation
- [x] Release PTQ implementations for AWQ, GPTQ, and SmoothQuant
- [ ] Unit tests
- [ ] Training implementation + full NVFP4 linear layer with 4/6

## Contributing

We welcome contributions to our repository, but get in touch before making any substantial changes.
Also, please make sure any code changes are compliant with our linter:

```bash
ruff check
```

## Citation

Please use the following BibTeX entry to cite this work:

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
```

## License

This repository is available under the MIT license.
See the [LICENSE.md](/LICENSE.md) file for details.
