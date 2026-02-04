from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...resources import FOUROVERSIX_CACHE_PATH, app, cache_volume, hf_secret
from ..experiment import Experiment
from ..utils import PTQMethod
from .rtn import RTNEvaluatorImpl, rtn_img

if TYPE_CHECKING:
    from fouroversix.utils import AdaptiveBlockScalingRule, DataType
    from sqlalchemy.orm import Session


with rtn_img.imports():
    import torch
    from fouroversix import fp4_matmul, quantize_model
    from fouroversix.model import FP4Linear
    from transformers import AutoModelForCausalLM


ALPHA_CANDIDATES = [x / 10 for x in range(11)]
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
                    "scale_rule": self.activation_scale_rule,
                    "fp4_format": self.fp4_format,
                },
                other_quantize_kwargs={
                    "scale_rule": self.weight_scale_rule,
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
        session: Session,
        **kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Get the kwargs for tasks that should be used to calibrate the given model for
        this PTQ method before running evaluation.
        """

        smoothquant_alpha = get_smoothquant_alpha(
            model_name,
            kwargs.get("activation_scale_rule"),
            kwargs.get("weight_scale_rule"),
            session,
        )

        calibration_experiments = get_calibration_experiments(
            model_name,
            kwargs.get("activation_scale_rule"),
            kwargs.get("weight_scale_rule"),
            session,
        )

        if smoothquant_alpha is None:
            return [
                {
                    "smoothquant_alpha": candidate_alpha,
                    "tasks": [WIKITEXT_TRAIN],
                }
                for candidate_alpha in ALPHA_CANDIDATES
                if not any(
                    experiment.smoothquant_alpha == candidate_alpha
                    for experiment in calibration_experiments
                )
            ]

        return []

    @classmethod
    def get_calibrated_kwargs(
        cls,
        model_name: str,
        session: Session,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Get the calibrated kwargs for the given model and scale rules. If this model
        has not yet been calibrated with these scale rules, an error will be raised.
        """

        smoothquant_alpha = get_smoothquant_alpha(
            model_name,
            kwargs.get("activation_scale_rule"),
            kwargs.get("weight_scale_rule"),
            session,
        )

        if smoothquant_alpha is None:
            msg = (
                "SmoothQuant has not been calibrated for this combination of model and "
                "scale rules"
            )
            raise ValueError(msg)

        return {"smoothquant_alpha": smoothquant_alpha}

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


def get_calibration_experiments(
    model_name: str,
    activation_scale_rule: AdaptiveBlockScalingRule,
    weight_scale_rule: AdaptiveBlockScalingRule,
    db_session: Session,
) -> list[Experiment]:
    return (
        db_session.query(Experiment)
        .filter(
            Experiment.ptq_method == PTQMethod.smoothquant.value,
            Experiment.task == WIKITEXT_TRAIN,
            Experiment.model_name == model_name,
            Experiment.activation_scale_rule == activation_scale_rule.value,
            Experiment.weight_scale_rule == weight_scale_rule.value,
            Experiment.smoothquant_alpha.isnot(None),
        )
        .all()
    )


def get_smoothquant_alpha(
    model_name: str,
    activation_scale_rule: AdaptiveBlockScalingRule,
    weight_scale_rule: AdaptiveBlockScalingRule,
    session: Session,
) -> float | None:
    calibration_experiments = get_calibration_experiments(
        model_name,
        activation_scale_rule,
        weight_scale_rule,
        session,
    )

    if not all(
        any(
            experiment.smoothquant_alpha == alpha
            for experiment in calibration_experiments
        )
        for alpha in ALPHA_CANDIDATES
    ):
        return None

    return min(calibration_experiments, key=lambda x: x.metric_value).smoothquant_alpha
