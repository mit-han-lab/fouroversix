from typing import Any

import click
import modal
from fouroversix import MatmulBackend, QuantizeBackend
from fouroversix.utils import AdaptiveBlockScalingRule, DataType, FP4Format

from ..resources import app
from .coordinators import LocalEvaluationCoordinator, ModalEvaluationCoordinator
from .utils import EvaluationFramework, PTQMethod


@click.command()
@click.option(
    "--a-scale-rule",
    type=AdaptiveBlockScalingRule,
    default=AdaptiveBlockScalingRule.mse,
)
@click.option("--detach", is_flag=True)
@click.option("--device", type=str, default="cuda")
@click.option("--dtype", type=DataType, default=DataType.auto)
@click.option(
    "--eval-framework",
    "-f",
    type=EvaluationFramework,
    default=EvaluationFramework.lm_eval,
)
@click.option("--fp4-format", type=FP4Format, default=FP4Format.nvfp4)
@click.option("--group-name", type=str, default=None)
@click.option("--limit", type=int, default=None)
@click.option("--matmul-backend", type=MatmulBackend, default=None)
@click.option("--max-length", type=int, default=None)
@click.option("--modal", is_flag=True)
@click.option("--modal-gpu", type=str)
@click.option("--model-name", "-m", type=str, multiple=True, required=True)
@click.option("--ptq-method", "-p", type=PTQMethod, multiple=True, required=True)
@click.option("--quantize-backend", type=QuantizeBackend, default=None)
@click.option("--task", "-t", type=str, multiple=True, default=["wikitext"])
@click.option("--trust-remote-code", is_flag=True)
@click.option(
    "--w-scale-rule",
    type=AdaptiveBlockScalingRule,
    default=AdaptiveBlockScalingRule.mse,
)
@click.option("--weight-scale-2d", is_flag=True)
def cli(group_name: str | None, **kwargs: dict[str, Any]) -> None:
    detach = kwargs.pop("detach", False)
    model_names = kwargs.pop("model_name")
    ptq_methods = kwargs.pop("ptq_method")
    use_modal = kwargs.pop("modal", False)
    kwargs["tasks"] = kwargs.pop("task")

    # Expand shortcuts
    if model_names[0] == "llamaqwen":
        model_names = [
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-70B",
            "Qwen/Qwen3-1.7B",
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3-32B",
        ]

    if isinstance(kwargs.get("tasks"), tuple):
        kwargs["tasks"] = list(kwargs.get("tasks"))

    # Validate options
    for ptq_method in ptq_methods:
        if ptq_method != PTQMethod.rtn:
            if kwargs.get("fp4_format") == FP4Format.mxfp4:
                msg = "MXFP4 is only supported with RTN"
                raise ValueError(msg)

            if kwargs.get("weight_scale_2d"):
                msg = "2D weight scales are only supported with RTN"
                raise ValueError(msg)

    if use_modal:
        with modal.enable_output(), app.run(detach=detach):
            coordinator = ModalEvaluationCoordinator(group_name_str=group_name or "")
            coordinator.start.remote(model_names, ptq_methods, **kwargs)
    else:
        coordinator = LocalEvaluationCoordinator(group_name)
        coordinator.start(model_names, ptq_methods, **kwargs)


if __name__ == "__main__":
    cli()
