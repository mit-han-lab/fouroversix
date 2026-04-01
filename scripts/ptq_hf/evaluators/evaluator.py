from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import modal
import torch
from fouroversix import (
    DataType,
    MatmulBackend,
    ModelQuantizationConfig,
    QuantizeBackend,
    ScaleRule,
)

from ..utils import EvaluationFramework

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from transformers import AutoConfig, AutoModelForCausalLM


class PTQEvaluator(ABC):
    """Base class for post-training quantization evaluators."""

    @classmethod
    def get_calibration_tasks(
        cls,
        model_name: str,  # noqa: ARG003
        session: Session,  # noqa: ARG003
        **kwargs: dict[str, Any],  # noqa: ARG003
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
        session: Session,  # noqa: ARG003
        **kwargs: dict[str, Any],  # noqa: ARG003
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
        dtype: DataType,
        eval_framework: EvaluationFramework,
        force_requantize_model: bool = False,
        limit: int | None,
        max_length: int | None,
        tasks: list[str],
        trust_remote_code: bool = False,
        matmul_backend: MatmulBackend | None = None,
        quantize_backend: QuantizeBackend | None = None,
        activation_dtype: DataType | None = None,
        activation_scale_rule: ScaleRule | None = None,
        scale_rule: ScaleRule | None = None,
        weight_dtype: DataType | None = None,
        weight_scale_2d: bool = False,
        weight_scale_rule: ScaleRule | None = None,
        save_path: Path | None = None,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a quantized model with lm-eval."""

        with torch.inference_mode():
            model_config = AutoConfig.from_pretrained(model_name)
            quantization_config = ModelQuantizationConfig(
                activation_dtype=activation_dtype,
                activation_scale_rule=activation_scale_rule,
                dtype=dtype,
                matmul_backend=matmul_backend,
                output_dtype=DataType(
                    (
                        str(model_config.dtype).replace("torch.", "")
                        if model_config.dtype is not None
                        else "bfloat16"
                    ),
                ),
                quantize_backend=quantize_backend,
                scale_rule=scale_rule,
                weight_scale_2d=weight_scale_2d,
                weight_dtype=weight_dtype,
                weight_scale_rule=weight_scale_rule,
            )

            # When running locally with --gpus-per-worker > 1, device will be a
            # string like "cuda:0,1,2,3", but this is not a valid torch device.
            # Here we create a max_memory dict to pass to from_pretrained and save
            # the first GPU as the main device.
            if "," in device:
                from_pretrained_kwargs = {
                    "device_map": "auto",
                    # Limit the memory usage of the first GPU to 80% of the
                    # device's free memory since we'll be letting lm-eval use this
                    # GPU to load tokens and etc.
                    "max_memory": {
                        i: (
                            torch.cuda.mem_get_info(i)[0]
                            * (0.8 if i == int(device.replace("cuda:", "")[0]) else 1)
                            if str(i) in device
                            else 0
                        )
                        for i in range(torch.cuda.device_count())
                    },
                    "trust_remote_code": trust_remote_code,
                }

                device = device.split(",", maxsplit=1)[0]
            else:
                from_pretrained_kwargs = {
                    "device_map": device,
                    "trust_remote_code": trust_remote_code,
                }

            model = self.quantize_model(
                model_name=model_name,
                force_requantize_model=force_requantize_model,
                save_path=save_path,
                quantization_config=quantization_config,
                from_pretrained_kwargs=from_pretrained_kwargs,
                **kwargs,
            )

            if eval_framework == EvaluationFramework.lm_eval:
                from lm_eval import evaluator
                from lm_eval.models.huggingface import HFLM
                from lm_eval.tasks import TaskManager

                eval_kwargs = {
                    "model": HFLM(
                        pretrained=model,
                        device=device,
                        max_length=max_length,
                    ),
                }

                full_results = evaluator.simple_evaluate(
                    tasks=tasks,
                    device=device,
                    limit=limit,
                    task_manager=TaskManager(
                        include_path=(
                            Path(__file__).parent.parent / "tasks"
                        ).as_posix(),
                    ),
                    **eval_kwargs,
                )

                results = []

                for task in full_results["results"]:
                    result = full_results["results"][task]

                    if "acc_norm,none" in result:
                        metric_name = "acc_norm,none"
                    elif "acc,none" in result:
                        metric_name = "acc,none"
                    elif "word_perplexity,none" in result:
                        metric_name = "word_perplexity,none"
                    else:
                        metric_name = None

                    results.append(
                        (
                            task,
                            metric_name,
                            result.get(metric_name),
                            full_results["results"][task],
                        ),
                    )

            elif eval_framework == EvaluationFramework.inspect_ai:
                import inspect_ai
                from inspect_ai.model import Model
                from inspect_ai.model._generate_config import GenerateConfig

                from .utils import local_hf

                config = GenerateConfig()
                full_results = inspect_ai.eval(
                    tasks=tasks,
                    model=Model(local_hf(model_name, model, config), config, None),
                    limit=limit,
                    log_dir=(save_path / "inspect_ai_logs").as_posix(),
                    log_realtime=True,
                    display="none",
                )

                results = []

                for log in full_results:
                    metrics = {
                        k: v.value
                        for score in log.results.scores
                        for k, v in score.metrics.items()
                    }

                    metric_name = "accuracy" if "accuracy" in metrics else None

                    results.append(
                        (
                            log.eval.task,
                            metric_name,
                            metrics.get(metric_name),
                            metrics,
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
        results = self.evaluate(*args, **kwargs)
        print(results)
        return results
