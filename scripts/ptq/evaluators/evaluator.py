from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import modal
import torch

if TYPE_CHECKING:
    from fouroversix import AdaptiveBlockScalingRule
    from sqlalchemy.orm import Session
    from transformers import AutoModelForCausalLM


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles torch.dtype."""

    def default(self, obj: Any) -> Any:  # noqa: ANN401
        """Convert value to a JSON serializable type."""

        import torch

        if isinstance(obj, torch.dtype):
            return str(obj)

        return json.JSONEncoder.default(self, obj)


class PTQEvaluator(ABC):
    """Base class for post-training quantization evaluators."""

    @classmethod
    def get_calibration_tasks(
        cls,
        model_name: str,  # noqa: ARG003
        a_scale_rule: AdaptiveBlockScalingRule,  # noqa: ARG003
        w_scale_rule: AdaptiveBlockScalingRule,  # noqa: ARG003
        save_path: Path,  # noqa: ARG003
    ) -> list[dict[str, Any]]:
        """
        Get the kwargs for tasks that should be used to calibrate the given model for
        this PTQ method before running evaluation.
        """
        return []

    @classmethod
    def get_calibrated_kwargs(
        cls,
        model_name: str,  # noqa: ARG003
        a_scale_rule: AdaptiveBlockScalingRule,  # noqa: ARG003
        w_scale_rule: AdaptiveBlockScalingRule,  # noqa: ARG003
        db_session: Session,  # noqa: ARG003
    ) -> dict[str, Any]:
        """
        Get the calibrated kwargs for the given model and scale rules. If this model
        has not yet been calibrated with these scale rules, an error will be raised.
        """
        return {}

    @abstractmethod
    def quantize_model(self, **kwargs: dict[str, Any]) -> AutoModelForCausalLM:
        """Quantize a model."""

    def evaluate(
        self,
        model_name: str,
        *,
        device: str,
        dtype: str,
        max_length: int,
        tasks: list[str],
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

        results = evaluator.simple_evaluate(
            model=models.huggingface.HFLM(
                pretrained=model,
                device=device,
                max_length=max_length,
            ),
            tasks=tasks,
            device=device,
            task_manager=TaskManager(
                include_path=(Path(__file__).parent.parent / "tasks").as_posix(),
            ),
        )

        del model
        torch.cuda.empty_cache()

        return results

    @modal.method()
    def evaluate_on_modal(
        self,
        *args: list[Any],
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a quantized model on Modal."""

        return self.evaluate(*args, **kwargs)
