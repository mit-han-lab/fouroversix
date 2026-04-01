import itertools
from pathlib import Path
from typing import Any

import modal

from ...resources import FOUROVERSIX_CACHE_PATH, app, cache_volume, get_image
from ..evaluators import get_evaluator
from ..utils import PTQMethod
from .base import BaseEvaluationCoordinator


@app.cls(
    image=get_image(),
    timeout=24 * 60 * 60,
    nonpreemptible=True,
    volumes={FOUROVERSIX_CACHE_PATH: cache_volume},
)
class ModalEvaluationCoordinator(BaseEvaluationCoordinator):
    """Evaluation coordinator for running PTQ experiments on Modal."""

    database_path_str: str = modal.parameter()
    group_name_str: str = modal.parameter()

    @property
    def database_path(self) -> Path:
        """Path to the SQLite database where experiment results are stored."""
        return FOUROVERSIX_CACHE_PATH / (
            self.database_path_str if self.database_path_str != "" else "results.db"
        )

    @property
    def group_name(self) -> str | None:
        """
        The name of the group experiments are being run in. If this is not None and an
        experiment with this group name and matching parameters has already been run,
        the experiment will not be run again.
        """

        # Modal doesn't allow None parameters in modal.parameter()
        return self.group_name_str if self.group_name_str != "" else None

    def run_calibration_tasks(
        self,
        model_names: list[str],
        ptq_methods: list[PTQMethod],
        tasks: list[str],
        modal_gpu: str,
        save_path: Path | None,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Run any tasks that should be used to calibrate models for a given PTQ method
        and set of parameters before running evaluation.
        """

        function_calls_with_inputs = []

        for model_name, ptq_method in itertools.product(model_names, ptq_methods):
            tasks_to_evaluate = self.get_tasks_to_evaluate(
                model_name,
                ptq_method,
                tasks,
            )

            if len(tasks_to_evaluate) == 0:
                continue

            evaluator_cls = get_evaluator(ptq_method).with_options(gpu=modal_gpu)

            function_calls_with_inputs.extend(
                [
                    (
                        model_name,
                        ptq_method,
                        {**kwargs, **calibration_task_kwargs},
                        evaluator_cls().evaluate_on_modal.spawn(
                            model_name=model_name,
                            save_path=FOUROVERSIX_CACHE_PATH / (save_path or "ptq"),
                            **{
                                **kwargs,
                                "tasks": tasks_to_evaluate,
                                **calibration_task_kwargs,
                            },
                        ),
                    )
                    for calibration_task_kwargs in evaluator_cls.get_calibration_tasks(
                        model_name,
                        self.get_session(),
                        **kwargs,
                    )
                ],
            )

        results = modal.FunctionCall.gather(
            *[function_call for _, _, _, function_call in function_calls_with_inputs],
        )

        for (model_name, ptq_method, function_call_kwargs, _), result in zip(
            function_calls_with_inputs,
            results,
            strict=True,
        ):
            self.save_results(model_name, ptq_method, function_call_kwargs, result)

    @modal.method()
    def start(
        self,
        model_names: list[str],
        ptq_methods: list[PTQMethod],
        tasks: list[str],
        *,
        gpus_per_worker: int,
        modal_gpu: str,
        run_tasks_in_parallel: bool,
        save_path: Path | None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Start the evaluation coordinator."""

        if gpus_per_worker is not None:
            msg = (
                "--gpus-per-worker is not supported when running on Modal. "
                "Use --modal-gpu instead."
            )
            raise ValueError(msg)

        self.run_calibration_tasks(
            model_names,
            ptq_methods,
            tasks,
            modal_gpu,
            save_path,
            **kwargs,
        )

        models_and_ptq_methods = list(itertools.product(model_names, ptq_methods))
        function_calls = []

        for model_name, ptq_method in models_and_ptq_methods:
            tasks_to_evaluate = self.get_tasks_to_evaluate(
                model_name,
                ptq_method,
                tasks,
            )

            if len(tasks_to_evaluate) == 0:
                continue

            evaluator_cls = get_evaluator(ptq_method).with_options(gpu=modal_gpu)

            calibrated_kwargs = evaluator_cls.get_calibrated_kwargs(
                model_name,
                self.get_session(),
                **kwargs,
            )

            if run_tasks_in_parallel:
                function_calls.extend(
                    [
                        evaluator_cls().evaluate_on_modal.spawn(
                            model_name=model_name,
                            tasks=[task_to_evaluate],
                            save_path=FOUROVERSIX_CACHE_PATH / (save_path or "ptq"),
                            **{**kwargs, **calibrated_kwargs},
                        )
                        for task_to_evaluate in tasks_to_evaluate
                    ],
                )
            else:
                function_calls.append(
                    evaluator_cls().evaluate_on_modal.spawn(
                        model_name=model_name,
                        tasks=tasks_to_evaluate,
                        save_path=FOUROVERSIX_CACHE_PATH / (save_path or "ptq"),
                        **{**kwargs, **calibrated_kwargs},
                    ),
                )

        all_results = modal.FunctionCall.gather(*function_calls)

        for (model_name, ptq_method), results in zip(
            models_and_ptq_methods,
            all_results,
            strict=True,
        ):
            self.save_results(model_name, ptq_method, kwargs, results)
