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
    from pathlib import Path

    from transformers import AutoModelForCausalLM


rtn_img = get_image()

with rtn_img.imports():
    from fouroversix import (
        DataType,
        FourOverSixLinearConfig,
        MatmulBackend,
        QuantizeBackend,
        ScaleRule,
        quantize_model,
    )
    from transformers import AutoModelForCausalLM


class RTNEvaluatorImpl(PTQEvaluator):
    """Evaluate a model using round-to-nearest quantization."""

    def quantize_model(
        self,
        model_name: str,
        *,
        device: str,
        dtype: DataType,
        save_path: Path,
        activation_scale_rule: ScaleRule,
        weight_scale_rule: ScaleRule,
        matmul_backend: MatmulBackend,
        quantize_backend: QuantizeBackend,
        weight_scale_2d: bool,
        model_kwargs: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> AutoModelForCausalLM:
        """Quantize a model using round-to-nearest quantization."""

        model_save_path = (
            save_path
            / "rtn"
            / f"{model_name}-{activation_scale_rule.value}-{weight_scale_rule.value}"
        )

        if not model_save_path.exists():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                **(model_kwargs or {}),
            )

            linear_config = FourOverSixLinearConfig(
                dtype=dtype,
                matmul_backend=matmul_backend,
                quantize_backend=quantize_backend,
                output_dtype=DataType(
                    str(model.config.torch_dtype).replace("torch.", ""),
                ),
                weight_scale_2d=weight_scale_2d,
            )

            linear_config.activation_scale_rule = activation_scale_rule
            linear_config.weight_scale_rule = weight_scale_rule

            quantize_model(model, linear_config=linear_config)
            # model.save_pretrained(model_save_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_save_path)
            model.name_or_path = model_name

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
