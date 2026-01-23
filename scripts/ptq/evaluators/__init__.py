from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..utils import PTQMethod
from .awq import AWQEvaluator
from .gptq import GPTQEvaluator
from .high_precision import HighPrecisionEvaluator
from .rtn import RTNEvaluator
from .smoothquant import (
    SmoothQuantAutoAlphaEvaluator,
    SmoothQuantEvaluator,
    get_smoothquant_alpha,
)
from .spinquant import SpinQuantEvaluationCoordinator

if TYPE_CHECKING:
    from .evaluator import PTQEvaluator


def get_evaluator(  # noqa: PLR0911
    ptq_method: PTQMethod,
    **kwargs: dict[str, Any],
) -> tuple[type[PTQEvaluator], dict[str, Any]]:
    """Get the evaluator class for the given PTQ method."""

    if ptq_method == PTQMethod.awq:
        return AWQEvaluator, {}
    if ptq_method == PTQMethod.gptq:
        return GPTQEvaluator, {}
    if ptq_method == PTQMethod.high_precision:
        return HighPrecisionEvaluator, {}
    if ptq_method == PTQMethod.rtn:
        return RTNEvaluator, {}
    if ptq_method == PTQMethod.smoothquant:
        smoothquant_alpha = get_smoothquant_alpha(
            kwargs["model_name"],
            kwargs["a_scale_rule"],
            kwargs["w_scale_rule"],
        )

        if smoothquant_alpha is None:
            return SmoothQuantAutoAlphaEvaluator, {}

        return SmoothQuantEvaluator, {"smoothquant_alpha": smoothquant_alpha}
    if ptq_method == PTQMethod.spinquant:
        return SpinQuantEvaluationCoordinator, {}

    msg = f"Unsupported PTQ method: {ptq_method}"
    raise ValueError(msg)
