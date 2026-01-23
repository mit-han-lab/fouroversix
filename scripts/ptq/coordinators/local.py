import itertools
import multiprocessing
from pathlib import Path
from typing import Any

import torch

from ..evaluators import get_evaluator
from ..utils import PTQMethod
from .base import BaseEvaluationCoordinator


class LocalEvaluationCoordinator(BaseEvaluationCoordinator):
    """Evaluation coordinator for running PTQ experiments locally."""

    def __init__(self, group_name: str | None = None) -> None:
        # Save the database in the fouroversix root directory
        self.database_path = Path(__file__).parent.parent.parent.parent / "results.db"
        self.group_name = group_name

    def evaluate(
        self,
        model_name: str,
        ptq_method: PTQMethod,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a model with a given PTQ method."""

        evaluator_cls, evaluator_kwargs = get_evaluator(
            ptq_method,
            model_name=model_name,
            **kwargs,
        )

        return evaluator_cls().evaluate(
            model_name=model_name,
            ptq_method=ptq_method,
            **evaluator_kwargs,
            **kwargs,
        )

    def start(
        self,
        model_names: list[str],
        ptq_methods: list[PTQMethod],
        tasks: list[str],
        device: str,
        **kwargs: dict[str, Any],
    ) -> None:
        """Start the evaluation coordinator."""

        multiprocessing.set_start_method("spawn", force=True)

        manager = multiprocessing.Manager()
        task_queue = manager.Queue()
        result_queue = manager.Queue()

        # Start one worker per GPU
        num_workers = max(torch.cuda.device_count(), 1)
        workers = []

        for gpu_id in range(num_workers):
            p = multiprocessing.Process(
                target=self.worker,
                args=(
                    f"cuda:{gpu_id}" if device == "cuda" else device,
                    task_queue,
                    result_queue,
                ),
            )
            p.start()
            workers.append(p)

        experiments = 0

        for model_name, ptq_method in itertools.product(model_names, ptq_methods):
            tasks_to_evaluate = self.get_tasks_to_evaluate(
                model_name,
                ptq_method,
                tasks,
            )

            if len(tasks_to_evaluate) == 0:
                continue

            task_queue.put(
                (model_name, ptq_method, {**kwargs, "tasks": tasks_to_evaluate}),
            )
            experiments += 1

        # Send shutdown signals (one per worker)
        for _ in range(num_workers):
            task_queue.put(None)

        # Collect results
        for _ in range(experiments):
            self.save_results(*result_queue.get())

        for p in workers:
            p.join()

    def worker(
        self,
        device: str,
        task_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
    ) -> None:
        """Worker process for running PTQ experiments locally."""

        while True:
            task = task_queue.get()

            if task is None:
                break

            model_name, ptq_method, kwargs = task
            result = self.evaluate(
                model_name,
                ptq_method,
                **{**kwargs, "device": device},
            )
            result_queue.put((model_name, ptq_method, result))
