from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from fouroversix.model import FP4Linear

from ...resources import (
    FOUROVERSIX_CACHE_PATH,
    Dependency,
    app,
    cache_volume,
    get_image,
    hf_secret,
)
from .rtn import RTNEvaluatorImpl

if TYPE_CHECKING:
    from fouroversix.utils import AdaptiveBlockScalingRule, DataType
    from transformers import AutoModelForCausalLM

awq_img = get_image(dependencies=[Dependency.fouroversix, Dependency.awq])


class FP4LinearForAWQ(FP4Linear):
    """
    Drop-in replacement for `FP4Linear` that quantizes the weights and activations
    during AWQ calibration.
    """

    def __init__(self, *args: list[Any], **kwargs: dict[str, Any]) -> None:
        super().__init__(*args, **kwargs)
        self.high_precision = False

    def apply_ptq(self) -> None:
        """
        Override the parent method to do nothing, since we need the high-precision
        weight when calibrating with AWQ.
        """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that can optionally be run in high precision. This is used to
        calculate the high-precision output to compare to during the auto-scale process
        in AWQ calibration.
        """

        return (
            super().forward(input)
            if self.high_precision
            else F.linear(input, self.weight, self.bias)
        )


@app.cls(
    image=awq_img,
    gpu="B200",
    secrets=[hf_secret],
    timeout=24 * 60 * 60,
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
)
class AWQEvaluator(RTNEvaluatorImpl):
    """Evaluate a model using AWQ."""

    def quantize_model(
        self,
        model_name: str,
        *,
        device: str,
        dtype: DataType,
        a_scale_rule: AdaptiveBlockScalingRule,
        w_scale_rule: AdaptiveBlockScalingRule,
        save_path: Path,
        model_kwargs: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ) -> AutoModelForCausalLM:
        """Quantize a model using AWQ."""

        import torch
        from awq.quantize.pre_quant import apply_awq, run_awq
        from fouroversix import quantize_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        save_path = (
            save_path
            / "awq"
            / f"{model_name}-{a_scale_rule.value}-{w_scale_rule.value}"
        )

        if not save_path.exists():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                dtype=dtype.torch(),
                **(model_kwargs or {}),
            ).eval()

            quantize_model(
                model,
                a_scale_rule=a_scale_rule,
                w_scale_rule=w_scale_rule,
                linear_cls=FP4LinearForAWQ,
                **kwargs,
            )

            enc = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                trust_remote_code=True,
            )

            awq_results = run_awq(
                model,
                enc,
                w_bit=16,
                q_config={"q_group_size": -1, "zero_point": False},
                n_samples=128,
                seqlen=512,
                calib_data="wikitext",
            )

            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(awq_results, save_path)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            dtype=dtype.torch(),
            **(model_kwargs or {}),
        )

        # Apply AWQ
        awq_results = torch.load(save_path, map_location="cuda")
        apply_awq(model, awq_results)

        # Quantize the model
        quantize_model(
            model,
            a_scale_rule=a_scale_rule,
            w_scale_rule=w_scale_rule,
            **kwargs,
        )

        return model.to(device)
