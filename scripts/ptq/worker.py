from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dateutil.tz import tzlocal

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
from .utils import PTQMethod

if TYPE_CHECKING:
    import multiprocessing

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


def worker(gpu_id: str, task_queue: multiprocessing.Queue) -> dict[str, Any]:
    while True:
        task = task_queue.get()

        if task is None:
            break

        model_name, ptq_method, kwargs = task

        evaluator_cls, evaluator_kwargs = get_evaluator(
            ptq_method,
            model_name=model_name,
            **kwargs,
        )
        results = evaluator_cls().evaluate.local(
            model_name=model_name,
            ptq_method=ptq_method,
            **{
                **kwargs,
                **evaluator_kwargs,
                "device": f"cuda:{gpu_id}",
            },
        )

        logs_path = (
            Path("ptq_logs")
            / ptq_method.value
            / f"{datetime.now(tz=tzlocal()).strftime('%Y%m%d%H%M%S')}_{model_name}.json"
        )
        logs_path.parent.mkdir(parents=True, exist_ok=True)

        with logs_path.open("w") as f:
            json.dump(
                {
                    "model_name": model_name,
                    "ptq_method": ptq_method.value,
                    "kwargs": kwargs,
                    "results": results["results"],
                },
                f,
                indent=4,
            )

        print("Saved PTQ results to", logs_path)
