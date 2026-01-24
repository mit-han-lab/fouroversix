import multiprocessing
from abc import ABC, abstractmethod
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..experiment import Base, Experiment
from ..utils import PTQMethod


class BaseEvaluationCoordinator(ABC):
    """Base class for evaluation coordinators."""

    def get_session(self) -> Session:
        """Get an SQLAlchemy session for the SQLite database."""
        engine = create_engine(f"sqlite:///{self.database_path.absolute().as_posix()}")
        Base.metadata.create_all(engine)
        return sessionmaker(bind=engine)()

    def get_tasks_to_evaluate(
        self,
        model_name: str,
        ptq_method: PTQMethod,
        tasks: list[str],
    ) -> list[str]:
        """
        Get the tasks that should be evaluated. If a group name is set, tasks will only
        be evaluated if they have not yet been evaluated for this group name, model
        name, PTQ method, and task.
        """

        if self.group_name is None:
            return tasks

        session = self.get_session()
        experiments = (
            session.query(Experiment)
            .filter(
                Experiment.group_name == self.group_name,
                Experiment.model_name == model_name,
                Experiment.ptq_method == ptq_method.value,
                Experiment.task.in_(tasks),
            )
            .all()
        )

        return [
            task
            for task in tasks
            if task not in [experiment.task for experiment in experiments]
        ]

    @abstractmethod
    def run_calibration_tasks(
        self,
        model_names: list[str],
        ptq_methods: list[PTQMethod],
        tasks: list[str],
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Run any tasks that should be used to calibrate models for a given PTQ method
        and set of parameters before running evaluation.
        """

    def save_results(
        self,
        model_name: str,
        ptq_method: PTQMethod,
        kwargs: dict[str, Any],
        full_results: dict[str, Any],
    ) -> None:
        """Save the results of a PTQ experiment to the SQLite database."""

        session = self.get_session()

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

            if metric_name is not None:
                metric_value = result[metric_name]

            experiment = Experiment(
                group_name=self.group_name,
                model_name=model_name,
                task=task,
                metric_name=metric_name,
                metric_value=metric_value,
                ptq_method=ptq_method.value,
                a_scale_rule=kwargs.get("a_scale_rule"),
                w_scale_rule=kwargs.get("w_scale_rule"),
                smoothquant_alpha=kwargs.get("smoothquant_alpha"),
                results=result,
            )
            session.add(experiment)

            print(model_name, ptq_method, task)
            print(result)

        session.commit()

    @abstractmethod
    def start(
        self,
        model_names: list[str],
        ptq_methods: list[PTQMethod],
        tasks: list[str],
        **kwargs: dict[str, Any],
    ) -> None:
        """Start the evaluation coordinator."""
