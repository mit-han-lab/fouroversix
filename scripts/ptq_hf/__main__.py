from pathlib import Path
from typing import Any

import click
import modal
from fouroversix.utils import DataType, MatmulBackend, QuantizeBackend, ScaleRule

from ..resources import app
from .coordinators import LocalEvaluationCoordinator, ModalEvaluationCoordinator
from .utils import EvaluationFramework, PTQMethod


@click.command()
@click.option("--activation-dtype", "--a-dtype", type=DataType, default=None)
@click.option("--activation-scale-rule", "--a-scale-rule", type=ScaleRule)
@click.option("--db-path", type=Path, default=None)
@click.option("--detach", is_flag=True)
@click.option("--device", type=str, default="cuda")
@click.option("--dtype", type=DataType, default=DataType.nvfp4)
@click.option(
    "--eval-framework",
    type=EvaluationFramework,
    default=EvaluationFramework.lm_eval,
)
@click.option(
    "--force-requantize-model",
    is_flag=True,
    help=(
        "Requantize any models that have already been quantized. Any models that have "
        "already been quantized and saved to disk will be overwritten."
    ),
)
@click.option(
    "--gpus-per-worker",
    type=int,
    help=(
        "Number of GPUs to give to each worker when running locally. Defaults to "
        "None, which will create a single worker with all of the machine's GPUs."
    ),
)
@click.option("--group-name", type=str, default=None)
@click.option("--limit", type=int, default=None)
@click.option("--matmul-backend", type=MatmulBackend, default=None)
@click.option("--max-length", type=int, default=None)
@click.option("--modal", is_flag=True)
@click.option("--modal-gpu", type=str)
@click.option("--model-name", "-m", type=str, multiple=True, required=True)
@click.option("--ptq-method", "-p", type=PTQMethod, multiple=True, required=True)
@click.option("--quantize-backend", type=QuantizeBackend, default=None)
@click.option(
    "--run-tasks-in-parallel",
    is_flag=True,
    help=(
        "Run tasks in parallel in cases where many workers are available (see "
        "--gpus-per-worker and/or --modal)."
    ),
)
@click.option("--save-path", type=Path, default=None)
@click.option("--scale-rule", type=ScaleRule, default=ScaleRule.mse)
@click.option("--task", "-t", type=str, multiple=True, default=["wikitext"])
@click.option("--trust-remote-code", is_flag=True)
@click.option("--weight-dtype", "--w-dtype", type=DataType, default=None)
@click.option("--weight-scale-rule", "--w-scale-rule", type=ScaleRule)
@click.option("--weight-scale-2d", "--w-scale-2d", is_flag=True)
def cli(
    *,
    db_path: Path | None,
    detach: bool,
    group_name: str | None,
    modal_gpu: str,
    **kwargs: dict[str, Any],
) -> None:
    activation_scale_rule = kwargs.get("activation_scale_rule") or kwargs.get(
        "scale_rule",
    )
    dtype: DataType = kwargs.get("dtype")
    weight_scale_rule = kwargs.get("weight_scale_rule") or kwargs.get("scale_rule")

    model_names = kwargs.pop("model_name")
    ptq_methods = kwargs.pop("ptq_method")
    tasks = kwargs.pop("task")
    use_modal = kwargs.pop("modal")

    if isinstance(tasks, tuple):
        tasks = list(tasks)

    if (
        activation_scale_rule not in dtype.supported_scale_rules
        or weight_scale_rule not in dtype.supported_scale_rules
    ):
        msg = (
            f"Either your activation scale rule ({activation_scale_rule}) or weight "
            f"scale rule ({weight_scale_rule}) is incompatible with your dtype "
            f"({dtype}). Please select another dtype (e.g. --dtype nvfp4) or another "
            f"scale rule (e.g. --a-scale-rule/--w-scale-rule) out of the ones "
            f"supported by {dtype}: {dtype.supported_scale_rules}"
        )
        raise ValueError(msg)

    if use_modal:
        with modal.enable_output(), app.run(detach=detach):
            coordinator = ModalEvaluationCoordinator(
                database_path_str=db_path.as_posix() if db_path is not None else "",
                group_name_str=group_name or "",
            )
            coordinator.start.remote(
                model_names,
                ptq_methods,
                tasks,
                modal_gpu=modal_gpu,
                **kwargs,
            )
    else:
        coordinator = LocalEvaluationCoordinator(group_name, db_path)
        coordinator.start(model_names, ptq_methods, tasks, **kwargs)


if __name__ == "__main__":
    cli()
