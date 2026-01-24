from __future__ import annotations

from typing import TYPE_CHECKING, Any

import modal

from ...resources import FOUROVERSIX_CACHE_PATH, app, cache_volume, hf_secret
from ..experiment import Experiment
from .rtn import RTNEvaluatorImpl, rtn_img

if TYPE_CHECKING:
    from fouroversix.utils import AdaptiveBlockScalingRule, DataType
    from sqlalchemy.orm import Session


with rtn_img.imports():
    import torch
    from fouroversix import fp4_matmul, quantize_model
    from fouroversix.model import FP4Linear
    from transformers import AutoModelForCausalLM


WIKITEXT_TRAIN = "wikitext_train"


class FP4LinearWithSmoothing(FP4Linear):
    """Drop-in replacement for `FP4Linear` that implements SmoothQuant-style scaling."""

    def __init__(
        self,
        *args: list[Any],
        smoothquant_alpha: float,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(*args, **kwargs)
        self.smoothquant_alpha = smoothquant_alpha

    def apply_ptq(self) -> None:
        """
        Override the parent method to do nothing, since we need the high-precision
        weight when doing PTQ with SmoothQuant.
        """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass for the FP4 linear layer with SmoothQuant-style scaling."""

        out = torch.empty(
            *input.shape[:-1],
            self.weight.shape[0],
            device=input.device,
            dtype=self.out_dtype,
        )

        # Slow bmm
        for i in range(input.shape[0]):
            s = (input[i].abs().max(dim=0).values ** self.smoothquant_alpha) / (
                self.weight.abs().max(dim=0).values ** (1 - self.smoothquant_alpha)
            )

            out[i] = fp4_matmul(
                input[i] / s[None, :],
                self.weight * s[None, :],
                out_dtype=self.out_dtype,
                input_quantize_kwargs={
                    "scale_rule": self.a_scale_rule,
                    "fp4_format": self.fp4_format,
                },
                other_quantize_kwargs={
                    "scale_rule": self.w_scale_rule,
                    "fp4_format": self.fp4_format,
                },
            )

        if self.bias is not None:
            out = out + self.bias

        return out


@app.cls(
    image=rtn_img,
    gpu="B200",
    secrets=[hf_secret],
    timeout=24 * 60 * 60,
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
)
class SmoothQuantEvaluator(RTNEvaluatorImpl):
    """Evaluate a model using SmoothQuant."""

    @classmethod
    def get_calibration_tasks(
        cls,
        model_name: str,
        a_scale_rule: AdaptiveBlockScalingRule,
        w_scale_rule: AdaptiveBlockScalingRule,
        session: Session,
    ) -> list[dict[str, Any]]:
        """
        Get the kwargs for tasks that should be used to calibrate the given model for
        this PTQ method before running evaluation.
        """

        smoothquant_alpha = get_smoothquant_alpha(
            model_name,
            a_scale_rule,
            w_scale_rule,
            session,
        )

        if smoothquant_alpha is None:
            return [
                {"smoothquant_alpha": candidate_alpha, "tasks": [WIKITEXT_TRAIN]}
                for candidate_alpha in [x / 10 for x in range(11)]
            ]

        return []

    @classmethod
    def get_calibrated_kwargs(
        cls,
        model_name: str,
        a_scale_rule: AdaptiveBlockScalingRule,
        w_scale_rule: AdaptiveBlockScalingRule,
        db_session: Session,
    ) -> dict[str, Any]:
        """
        Get the calibrated kwargs for the given model and scale rules. If this model
        has not yet been calibrated with these scale rules, an error will be raised.
        """

        smoothquant_alpha = get_smoothquant_alpha(
            model_name,
            a_scale_rule,
            w_scale_rule,
            db_session,
        )

        if smoothquant_alpha is None:
            msg = (
                "SmoothQuant has not been calibrated for this combination of model and "
                "scale rules"
            )
            raise ValueError(msg)

        return {
            "smoothquant_alpha": smoothquant_alpha,
        }

    def quantize_model(
        self,
        model_name: str,
        *,
        device: str,
        dtype: DataType,
        smoothquant_alpha: float,
        model_kwargs: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ) -> AutoModelForCausalLM:
        """Quantize a model using SmoothQuant."""

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            dtype=dtype.torch(),
            **(model_kwargs or {}),
        )

        quantize_model(
            model,
            linear_cls=FP4LinearWithSmoothing,
            smoothquant_alpha=smoothquant_alpha,
            **kwargs,
        )

        return model

    @modal.method()
    def smoothquant_evaluate(
        self,
        model_name: str,
        smoothquant_alpha: float,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a model using SmoothQuant."""

        return super().evaluate_impl(
            model_name,
            smoothquant_alpha=smoothquant_alpha,
            **kwargs,
        )


def get_smoothquant_alpha(
    model_name: str,
    a_scale_rule: AdaptiveBlockScalingRule,
    w_scale_rule: AdaptiveBlockScalingRule,
    db_session: Session,
) -> float | None:
    experiments = (
        db_session.query(Experiment)
        .filter(
            Experiment.ptq_method == "smoothquant",
            Experiment.task == WIKITEXT_TRAIN,
            Experiment.model_name == model_name,
            Experiment.a_scale_rule == a_scale_rule,
            Experiment.w_scale_rule == w_scale_rule,
        )
        .all()
    )

    if len(experiments) == 0:
        return None

    return min(experiments, key=lambda x: x.metric_value).smoothquant_alpha
