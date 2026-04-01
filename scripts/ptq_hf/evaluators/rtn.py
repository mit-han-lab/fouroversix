from __future__ import annotations

from typing import TYPE_CHECKING, Any

import modal

from ...resources import FOUROVERSIX_CACHE_PATH, app, cache_volume, get_image, hf_secret
from .evaluator import PTQEvaluator

if TYPE_CHECKING:
    from pathlib import Path

    from fouroversix import ModelQuantizationConfig
    from transformers import AutoModelForCausalLM


rtn_img = get_image(
    extra_pip_dependencies=[
        "transformers @ git+https://github.com/kathrynle20/transformers.git@jack/dtypes",
    ],
)

with rtn_img.imports():
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        from transformers import FourOverSixConfig as HFFourOverSixConfig
    except ImportError:
        HFFourOverSixConfig = None


class RTNEvaluatorImpl(PTQEvaluator):
    """Evaluate a model using round-to-nearest quantization."""

    def quantize_model(
        self,
        model_name: str,
        *,
        force_requantize_model: bool,
        from_pretrained_kwargs: dict[str, Any],
        save_path: Path,
        quantization_config: ModelQuantizationConfig,
    ) -> AutoModelForCausalLM:
        """Quantize a model using round-to-nearest quantization."""

        model_save_path = (
            save_path / "rtn" / model_name / quantization_config.__hash__()
        )

        if force_requantize_model or not model_save_path.exists():
            model_config = AutoConfig.from_pretrained(model_name)

            hf_quantization_config = HFFourOverSixConfig(
                activation_dtype=quantization_config.activation_dtype,
                activation_scale_rule=quantization_config.activation_scale_rule,
                dtype=quantization_config.dtype,
                matmul_backend=quantization_config.matmul_backend,
                output_dtype=quantization_config.output_dtype,
                quantize_backend=quantization_config.quantize_backend,
                weight_dtype=quantization_config.weight_dtype,
                weight_scale_2d=quantization_config.weight_scale_2d,
                weight_scale_rule=quantization_config.weight_scale_rule,
            )

            save_kwargs = {}
            if hasattr(model_config, "quantization_config"):
                hf_quantization_config.pre_quantized_model_config_type = str(
                    type(model_config),
                )
                save_kwargs["save_original_format"] = False
                delattr(model_config, "quantization_config")

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=model_config,
                quantization_config=hf_quantization_config,
                **from_pretrained_kwargs,
            )

            if model.__class__.__name__ == "Qwen3_5ForCausalLM":
                model.config.model_type = model_config.model_type

            if hasattr(hf_quantization_config, "pre_quantized_model_config_type"):
                delattr(hf_quantization_config, "pre_quantized_model_config_type")

            model.save_pretrained(model_save_path, **save_kwargs)

            if not modal.is_local():
                cache_volume.commit()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_save_path,
                **from_pretrained_kwargs,
            )

        # Fix for Inspect AI
        model.name_or_path = model_name

        return model


@app.cls(
    image=rtn_img,
    cpu=4,
    memory=8 * 1024,
    gpu="B200",
    secrets=[hf_secret],
    timeout=24 * 60 * 60,
    scaledown_window=30,
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
)
class RTNEvaluator(RTNEvaluatorImpl):
    """Evaluate a model using round-to-nearest quantization."""
