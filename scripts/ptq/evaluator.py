from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import modal
from dateutil.tz import tzlocal

from ..resources import FOUROVERSIX_CACHE_PATH

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles torch.dtype."""

    def default(self, obj: Any) -> Any:  # noqa: ANN401
        """Convert value to a JSON serializable type."""

        import torch

        if isinstance(obj, torch.dtype):
            return str(obj)

        return json.JSONEncoder.default(self, obj)


class PTQEvaluatorImpl(ABC):
    """Base class for post-training quantization evaluators."""

    @abstractmethod
    def quantize_model(self, **kwargs: dict[str, Any]) -> AutoModelForCausalLM:
        """Quantize a model."""

    def evaluate_impl(
        self,
        model_name: str,
        *,
        device: str,
        dtype: str,
        tasks: list[str],
        limit: int | None = None,
        trust_remote_code: bool = False,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a quantized model with lm-eval."""

        from lm_eval import evaluator, models
        from lm_eval.tasks import TaskManager

        if isinstance(model_name, str):
            model = self.quantize_model(
                model_name=model_name,
                device=device,
                dtype=dtype,
                model_kwargs={"trust_remote_code": trust_remote_code},
                **kwargs,
            )
        else:
            model = model_name

        return evaluator.simple_evaluate(
            model=models.huggingface.HFLM(pretrained=model, device=device),
            tasks=tasks,
            device=device,
            limit=limit,
            task_manager=TaskManager(
                include_path=(Path(__file__).parent / "tasks").as_posix(),
            ),
        )


class PTQEvaluator(PTQEvaluatorImpl):
    """Base class for post-training quantization evaluators."""

    @modal.method()
    def evaluate(self, *args: list[str], **kwargs: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a quantized model."""
        model_name = kwargs["model_name"]
        ptq_method = kwargs["ptq_method"]

        results = self.evaluate_impl(*args, **kwargs)

        logs_path = (
            FOUROVERSIX_CACHE_PATH
            / "ptq_logs"
            / (
                f"{model_name}_{ptq_method.value}-{datetime.now(tz=tzlocal()).strftime('%Y%m%d%H%M%S')}.json"
            )
        )
        logs_path.parent.mkdir(parents=True, exist_ok=True)

        with logs_path.open("w") as f:
            json.dump(
                {
                    "model_name": model_name,
                    "ptq_method": ptq_method.value,
                    "kwargs": kwargs,
                    "results": results,
                },
                f,
                indent=4,
                cls=CustomJSONEncoder,
            )

        return results
