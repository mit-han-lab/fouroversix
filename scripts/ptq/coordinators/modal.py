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

    group_name_str: str = modal.parameter()

    @property
    def database_path(self) -> Path:
        """Path to the SQLite database where experiment results are stored."""
        return FOUROVERSIX_CACHE_PATH / "results.db"

    @property
    def group_name(self) -> str | None:
        """
        The name of the group experiments are being run in. If this is not None and an
        experiment with this group name and matching parameters has already been run,
        the experiment will not be run again.
        """

        # Modal doesn't allow None parameters in modal.parameter()
        return self.group_name_str if self.group_name_str != "" else None

    @modal.method()
    def start(
        self,
        model_names: list[str],
        ptq_methods: list[PTQMethod],
        tasks: list[str],
        **kwargs: dict[str, Any],
    ) -> None:
        """Start the evaluation coordinator."""

        function_calls = []

        for model_name, ptq_method in itertools.product(model_names, ptq_methods):
            tasks_to_evaluate = self.get_tasks_to_evaluate(
                model_name,
                ptq_method,
                tasks,
            )

            if len(tasks_to_evaluate) == 0:
                continue

            evaluator_cls = get_evaluator(ptq_method)

            function_calls.append(
                evaluator_cls().evaluate_on_modal.spawn(
                    model_name=model_name,
                    ptq_method=ptq_method,
                    tasks=tasks_to_evaluate,
                    save_path=FOUROVERSIX_CACHE_PATH / "ptq",
                    **kwargs,
                ),
            )

        results = modal.FunctionCall.gather(*function_calls)

        for (model_name, ptq_method), result in zip(
            itertools.product(model_names, ptq_methods),
            results,
            strict=True,
        ):
            self.save_results(model_name, ptq_method, kwargs, result)
