import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...resources import (
    FOUROVERSIX_CACHE_PATH,
    Dependency,
    app,
    cache_volume,
    get_image,
    hf_secret,
)
from .evaluator import PTQEvaluator

if TYPE_CHECKING:
    from fouroversix.utils import AdaptiveBlockScalingRule, DataType
    from transformers import AutoModelForCausalLM

CALIBRATION_DATASET = "wikitext"

gptq_img = get_image(
    dependencies=[
        Dependency.fast_hadamard_transform,
        Dependency.qutlass,
        Dependency.fouroversix,
        Dependency.fp_quant,
    ],
)


@app.cls(
    image=gptq_img,
    gpu="B200",
    secrets=[hf_secret],
    timeout=24 * 60 * 60,
    volumes={FOUROVERSIX_CACHE_PATH: cache_volume},
)
class GPTQEvaluator(PTQEvaluator):
    """Evaluate a model after quantizing it with GPTQ."""

    def quantize_model(
        self,
        model_name: str,
        *,
        device: str,
        dtype: "DataType",
        a_scale_rule: "AdaptiveBlockScalingRule",
        w_scale_rule: "AdaptiveBlockScalingRule",
        save_path: Path,
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> "AutoModelForCausalLM":
        """Quantize a model with GPTQ."""

        import fouroversix
        from fouroversix.utils import DataType

        sys.path.extend(
            [
                (
                    Path(fouroversix.__file__).parent.parent.parent
                    / "third_party"
                    / "fp-quant"
                ).as_posix(),
            ],
        )

        from model_quant import main
        from transformers import AutoModelForCausalLM

        if dtype == DataType.auto:
            dtype = DataType.bfloat16
            msg = (
                "GPTQ only supports bfloat16, dtype is currently set to auto. "
                "Switching to bfloat16..."
            )
            warnings.warn(msg, stacklevel=2)

        save_path = (
            save_path
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
