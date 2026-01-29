from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...resources import (
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
    from fouroversix import QuantizeBackend, quantize_model
    from transformers import AutoModelForCausalLM


class RTNEvaluatorImpl(PTQEvaluator):
    """Evaluate a model using round-to-nearest quantization."""

    def quantize_model(
        self,
        model_name: str,
        *,
        device: str,
        dtype: DataType,
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

        quantize_model(model, **kwargs)
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
