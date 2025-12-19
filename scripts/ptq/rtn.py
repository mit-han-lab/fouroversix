from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..resources import (
    FOUROVERSIX_CACHE_PATH,
    app,
    cache_volume,
    get_image,
    hf_secret,
)
from .evaluator import PTQEvaluator

if TYPE_CHECKING:
    from fouroversix.utils import DataType
    from transformers import AutoModelForCausalLM


rtn_img = get_image()

with rtn_img.imports():
    from fouroversix import QuantizeBackend, apply_ptq
    from transformers import AutoModelForCausalLM


class RTNEvaluatorImpl(PTQEvaluator):
    """Evaluate a model using round-to-nearest quantization."""

    def quantize_model(
        self,
        model_name: str,
        *,
        device: str,
        dtype: DataType,
        quantize_backend: QuantizeBackend | None = None,
        model_kwargs: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ) -> AutoModelForCausalLM:
        """Quantize a model using round-to-nearest quantization."""

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            dtype=dtype.torch(),
            **(model_kwargs or {}),
        )
        apply_ptq(
            model,
            device=device,
            dtype=dtype,
            a_quantize_kwargs={"backend": quantize_backend},
            w_quantize_kwargs={"backend": quantize_backend},
            **kwargs,
        )
        return model


@app.cls(
    image=rtn_img,
    cpu=4,
    memory=8 * 1024,
    gpu="B200",
    secrets=[hf_secret],
    timeout=24 * 60 * 60,
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
)
class RTNEvaluator(RTNEvaluatorImpl):
    """Evaluate a model using round-to-nearest quantization."""
