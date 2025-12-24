import multiprocessing
from typing import Any

import click
import modal
from fouroversix import MatmulBackend, QuantizeBackend
from fouroversix.utils import AdaptiveBlockScalingRule, DataType, FP4Format

from ..resources import app, get_image
from .utils import PTQMethod, print_results
from .worker import get_evaluator, worker


@app.function(
    image=get_image(),
    timeout=24 * 60 * 60,
    nonpreemptible=True,
)
def run_ptq(
    model_names: list[str],
    ptq_methods: list[PTQMethod],
    **kwargs: dict[str, Any],
) -> None:
    function_calls = []

    for model_name in model_names:
        for ptq_method in ptq_methods:
            evaluator_cls, evaluator_kwargs = get_evaluator(
                ptq_method,
                model_name=model_name,
                **kwargs,
            )
            function_calls.append(
                evaluator_cls().evaluate.spawn(
                    model_name=model_name,
                    ptq_method=ptq_method,
                    **{**kwargs, **evaluator_kwargs},
                ),
            )

    print("Starting PTQ evaluation...")
    print(kwargs)
    print()

    results = modal.FunctionCall.gather(*function_calls)

    for i, model_name in enumerate(model_names):
        for j, ptq_method in enumerate(ptq_methods):
            print(model_name, ptq_method)
            print_results(results[i * len(ptq_methods) + j])
            print()


@click.command()
@click.option(
    "--a-scale-rule",
    type=AdaptiveBlockScalingRule,
    default=AdaptiveBlockScalingRule.mse,
)
@click.option("--detach", is_flag=True)
@click.option("--device", type=str, default="cuda")
@click.option("--dtype", type=DataType, default=DataType.bfloat16)
@click.option("--fp4-format", type=FP4Format, default=FP4Format.nvfp4)
@click.option("--matmul-backend", type=MatmulBackend, default=None)
@click.option("--modal", is_flag=True)
@click.option("--model-name", "-m", type=str, multiple=True, required=True)
@click.option("--ptq-method", "-p", type=PTQMethod, multiple=True, required=True)
@click.option("--quantize-backend", type=QuantizeBackend, default=None)
@click.option("--tasks", "-t", type=str, multiple=True, default=["wikitext"])
@click.option("--trust-remote-code", is_flag=True)
@click.option(
    "--w-scale-rule",
    type=AdaptiveBlockScalingRule,
    default=AdaptiveBlockScalingRule.mse,
)
@click.option("--weight-scale-2d", is_flag=True)
def cli(**kwargs: dict[str, Any]) -> None:  # noqa: C901, PLR0912
    detach = kwargs.pop("detach", False)
    model_names = kwargs.pop("model_name")
    ptq_methods = kwargs.pop("ptq_method")
    use_modal = kwargs.pop("modal", False)

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

            if kwargs.get("quantize_backend") is not None:
                msg = "Setting the quantization backend is only supported with RTN"
                raise ValueError(msg)

            if kwargs.get("weight_scale_2d"):
                msg = "2D weight scales are only supported with RTN"
                raise ValueError(msg)

    if use_modal:
        with modal.enable_output(), app.run(detach=detach):
            run_ptq.remote(model_names, ptq_methods, **kwargs)
    else:
        import torch

        if not torch.cuda.is_available():
            msg = "No CUDA devices found"
            raise RuntimeError(msg)

        multiprocessing.set_start_method("spawn", force=True)

        task_queue = multiprocessing.Queue()
        workers = []

        for gpu_id in range(torch.cuda.device_count()):
            p = multiprocessing.Process(target=worker, args=(gpu_id, task_queue))
            p.start()
            workers.append(p)

        for model_name in model_names:
            for ptq_method in ptq_methods:
                task_queue.put((model_name, ptq_method, kwargs))

        for _ in range(torch.cuda.device_count()):
            task_queue.put(None)

        for p in workers:
            p.join()


if __name__ == "__main__":
    cli()
