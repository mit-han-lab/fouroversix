from __future__ import annotations

from typing import TYPE_CHECKING

from fouroversix import quantize_model

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
    from fouroversix import ModelQuantizationConfig
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
        device: str,
        save_path: Path,
        quantization_config: ModelQuantizationConfig,
        trust_remote_code: bool = False,
    ) -> AutoModelForCausalLM:
        """Quantize a model using round-to-nearest quantization."""

        model_save_path = (
            save_path
            / "rtn"
            / (
                f"{model_name}-{quantization_config.get_activation_scale_rule().value}"
                f"-{quantization_config.get_weight_scale_rule().value}"
            )
        )

        if not model_save_path.exists():
            model_config = AutoConfig.from_pretrained(model_name)

            if (
                hasattr(model_config, "quantization_config")
                or HFFourOverSixConfig is None
            ):
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device,
                    trust_remote_code=trust_remote_code,
                )

                quantize_model(model, quantization_config)
            else:
                hf_quantization_config = HFFourOverSixConfig(
                    activation_scale_rule=quantization_config.get_activation_scale_rule(),
                    dtype=quantization_config.dtype,
                    matmul_backend=quantization_config.matmul_backend,
                    output_dtype=quantization_config.output_dtype,
                    quantize_backend=quantization_config.quantize_backend,
                    weight_scale_2d=quantization_config.weight_scale_2d,
                    weight_scale_rule=quantization_config.get_weight_scale_rule(),
                )

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device,
                    quantization_config=hf_quantization_config,
                    trust_remote_code=trust_remote_code,
                )

                model.save_pretrained(model_save_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_save_path,
                device_map=device,
                trust_remote_code=trust_remote_code,
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
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
)
class RTNEvaluator(RTNEvaluatorImpl):
    """Evaluate a model using round-to-nearest quantization."""
