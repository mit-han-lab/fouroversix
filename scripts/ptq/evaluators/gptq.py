import sys
from typing import Any

from ...resources import (
    FOUROVERSIX_CACHE_PATH,
    FOUROVERSIX_INSTALL_PATH,
    Dependency,
    app,
    cache_volume,
    get_image,
    hf_secret,
)
from .evaluator import PTQEvaluator

CALIBRATION_DATASET = "wikitext"

gptq_img = get_image(
    dependencies=[
        Dependency.fast_hadamard_transform,
        Dependency.qutlass,
        Dependency.fouroversix,
        Dependency.fp_quant,
    ],
)

with gptq_img.imports():
    sys.path.extend(
        [
            f"{FOUROVERSIX_INSTALL_PATH}/fpquant",
            f"{FOUROVERSIX_INSTALL_PATH}/fpquant/fpquant_cli",
        ],
    )

    from fouroversix.utils import AdaptiveBlockScalingRule, DataType
    from fpquant_cli.model_quant import main
    from transformers import AutoModelForCausalLM


@app.cls(
    image=gptq_img,
    gpu="B200",
    secrets=[hf_secret],
    timeout=24 * 60 * 60,
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
)
class GPTQEvaluator(PTQEvaluator):
    """Evaluate a model after quantizing it with GPTQ."""

    def quantize_model(
        self,
        model_name: str,
        *,
        device: str,
        dtype: DataType,
        a_scale_rule: AdaptiveBlockScalingRule,
        w_scale_rule: AdaptiveBlockScalingRule,
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> "AutoModelForCausalLM":
        """Quantize a model with GPTQ."""

        save_path = (
            FOUROVERSIX_CACHE_PATH
            / "ptq"
            / "gptq"
            / (f"{model_name}-{a_scale_rule.value}-{w_scale_rule.value}")
        )

        if not save_path.exists():
            sys.argv = [
                *sys.argv,
                "--model_name_or_path",
                model_name,
                "--dataset_name_or_path",
                CALIBRATION_DATASET,
                "--w_bits",
                "4",
                "--a_bits",
                "4",
                "--export_quantized_model",
                "realquant",
                "--format",
                "nvfp",
                "--gptq",
                "--save_path",
                save_path.as_posix(),
                "--a_scale_selection_rule",
                a_scale_rule.value,
                "--w_scale_selection_rule",
                w_scale_rule.value,
            ]

            main()

        return AutoModelForCausalLM.from_pretrained(
            save_path,
            device_map=device,
            dtype=dtype.torch(),
        )
