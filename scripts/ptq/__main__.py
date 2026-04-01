import os
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
from typing import Any

import click
import modal
import requests
import torch
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import Session, sessionmaker

from ..resources import FOUROVERSIX_CACHE_PATH, app, cache_volume, get_image, hf_secret
from .experiment import Base, Experiment
from .utils import PTQMethod, QuantizationScheme, Task, TaskType

VLLM_PROCESS = None

ptq_img = get_image(
    run_before_copy=lambda img: img.run_commands(
        "pip install vllm==0.18.0",
        'pip install "llmcompressor @ git+https://github.com/vllm-project/llm-compressor.git"',
        'pip install "transformers @ git+https://github.com/kathrynle20/transformers.git@jack/dtypes"',
    ),
)


def cleanup(*_) -> None:  # noqa: ANN002
    global VLLM_PROCESS  # noqa: PLW0602

    if VLLM_PROCESS and VLLM_PROCESS.poll() is None:
        try:
            os.killpg(VLLM_PROCESS.pid, signal.SIGTERM)
            VLLM_PROCESS.wait(timeout=5)
        except:  # noqa: E722
            os.killpg(VLLM_PROCESS.pid, signal.SIGKILL)

    if modal.is_local():
        sys.exit(1)


def get_db_session() -> Session:
    engine = create_engine(
        (
            "sqlite:///results.db"
            if modal.is_local()
            else "sqlite:////fouroversix/results.db"
        ),
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def wait_for_server(url: str) -> None:
    url = f"{url}/health"
    print("Waiting for server...")
    while True:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:  # noqa: PLR2004
                return
        except:  # noqa: E722, S110
            pass
        time.sleep(1)


@app.function(
    image=ptq_img,
    gpu="B200:8",
    secrets=[hf_secret],
    timeout=24 * 60 * 60,
    scaledown_window=30,
    volumes={FOUROVERSIX_CACHE_PATH: cache_volume},
)
def run_ptq(  # noqa: C901, PLR0912, PLR0915
    *,
    enforce_eager: bool,
    ignore_existing_results: bool,
    limit: int,
    model: str,
    num_concurrent: int,
    num_repeats: int,
    port: int,
    ptq_method: PTQMethod,
    quantization_scheme: QuantizationScheme,
    rht: bool,
    tasks: list[str],
    vllm_server_url: str,
) -> list[dict[str, Any]]:
    # Print environment
    for key, value in locals().items():
        print(f"{key}: {value}")

    global VLLM_PROCESS  # noqa: PLW0603

    # Set up dgx
    if socket.gethostname().startswith("dgx"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    session = get_db_session()

    # Skip tasks that already have results
    skip_tasks = []

    for task in tasks:
        existing_experiments = (
            session.query(Experiment)
            .filter(
                Experiment.model == model,
                or_(
                    Experiment.task == task.value,
                    Experiment.task == task.inspect_name,
                ),
                Experiment.quantization_scheme
                == (
                    quantization_scheme.value
                    if quantization_scheme is not None
                    else None
                ),
                Experiment.ptq_method == ptq_method.value,
                Experiment.rht == rht,
            )
            .all()
        )

        if not ignore_existing_results and len(existing_experiments) > 0:
            print(f"Skipping {task} because it already has results")
            skip_tasks.append(task)
            continue

    tasks = [task for task in tasks if task not in skip_tasks]

    if len(tasks) == 0:
        print("No tasks to evaluate")
        return []

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    vllm_cmd = ["vllm", "serve", model, "--port", str(port), "--trust-remote-code"]

    if enforce_eager:
        vllm_cmd.append("--enforce-eager")

    if quantization_scheme is not None:
        vllm_cmd.extend(
            [
                "--quantization",
                quantization_scheme.vllm_quantization_name + ("_rht" if rht else ""),
            ],
        )

    if "Qwen3.5" in model:
        vllm_cmd.extend(
            [
                "--reasoning-parser",
                "qwen3",
                "--enable-prefix-caching",
                "--language-model-only",
            ],
        )

        if not enforce_eager:
            vllm_cmd.extend(
                ["--max-cudagraph-capture-size", "16"],
            )

        if torch.cuda.device_count() > 1:
            if re.search(r"\bA\d+B\b", model):
                vllm_cmd.extend(
                    [
                        "--tensor-parallel-size",
                        str(torch.cuda.device_count()),
                        "--enable-expert-parallel",
                    ],
                )
            else:
                vllm_cmd.extend(
                    ["--data-parallel-size", str(torch.cuda.device_count())],
                )
        if model.startswith("Qwen/Qwen3.5-397B-A17B"):
            vllm_cmd.extend(
                [
                    "--max-model-len",
                    "262144",
                ],
            )

    elif model.startswith("nvidia/NVIDIA-Nemotron-3"):
        vllm_cmd.extend(
            [
                "--max-model-len",
                "262144",
                "--reasoning-parser-plugin",
                "/home/cookj/nano_v3_reasoning_parser.py",
                "--reasoning-parser",
                "nano_v3",
            ],
        )

        if not enforce_eager:
            vllm_cmd.extend(
                ["--max-cudagraph-capture-size", "16"],
            )

        if torch.cuda.device_count() > 1:
            if re.search(r"\bA\d+B\b", model):
                vllm_cmd.extend(
                    [
                        "--tensor-parallel-size",
                        str(torch.cuda.device_count()),
                        "--enable-expert-parallel",
                    ],
                )
            else:
                vllm_cmd.extend(
                    ["--data-parallel-size", str(torch.cuda.device_count())],
                )
    elif model == "MiniMaxAI/MiniMax-M2.5":
        vllm_cmd.extend(
            [
                "--tensor-parallel-size",
                str(torch.cuda.device_count()),
                "--enable-expert-parallel",
                "--reasoning-parser",
                "minimax_m2_append_think",
            ],
        )

        if not enforce_eager:
            vllm_cmd.extend(
                ["--max-cudagraph-capture-size", "16"],
            )
    elif model.startswith("stepfun-ai/Step-3.5-Flash"):
        vllm_cmd.extend(
            [
                "--tensor-parallel-size",
                str(torch.cuda.device_count()),
                "--enable-expert-parallel",
                "--disable-cascade-attn",
                "--reasoning-parser",
                "step3p5",
                "--hf-overrides",
                '{"num_nextn_predict_layers": 1}',
                # "--tokenizer",
                # "stepfun-ai/Step-3.5-Flash",
                "--hf-config-path",
                # Fix max_position_embedding -> max_position_embeddings in their
                # config on Hugging Face
                "/home/cookj/Step-3.5-Flash-Base-config",
                "--trust-remote-code",
            ],
        )

        if not enforce_eager:
            vllm_cmd.extend(
                ["--max-cudagraph-capture-size", "16"],
            )
    elif model == "mistralai/Mistral-Small-4-119B-2603":
        vllm_cmd.extend(
            [
                "--tensor-parallel-size",
                str(torch.cuda.device_count()),
                "--reasoning-parser",
                "mistral",
                "--max-model-len",
                "262144",
            ],
        )

        if not enforce_eager:
            vllm_cmd.extend(
                ["--max-cudagraph-capture-size", "16"],
            )
    elif model == "meta-llama/Llama-3.1-70B":
        vllm_cmd.extend(
            [
                "--tensor-parallel-size",
                str(torch.cuda.device_count()),
            ],
        )

        if not enforce_eager:
            vllm_cmd.extend(
                ["--max-cudagraph-capture-size", "16"],
            )

    if vllm_server_url is None:
        VLLM_PROCESS = subprocess.Popen(vllm_cmd, start_new_session=True)
        vllm_server_url = f"http://localhost:{port}"

    results = []

    try:
        wait_for_server(vllm_server_url)

        for task in tasks:
            for repeat in range(num_repeats or task.num_repeats):
                if task.task_type in {TaskType.question_answering, TaskType.perplexity}:
                    from lm_eval import evaluator

                    try:
                        full_results = evaluator.simple_evaluate(
                            model="local-completions",
                            model_args={
                                "model": model,
                                "base_url": f"{vllm_server_url}/v1/completions",
                                "num_concurrent": num_concurrent,
                                "tokenized_requests": False,
                                **(
                                    {"tokenizer": "stepfun-ai/Step-3.5-Flash"}
                                    if model.startswith("stepfun-ai/Step-3.5-Flash")
                                    else {}
                                ),
                            },
                            tasks=[task.value],
                            limit=limit,
                            gen_kwargs={"temperature": 0.0},
                        )
                    except UnboundLocalError:
                        print(f"Task {task} failed!")
                        traceback.print_exc()
                        break

                    for subtask in full_results["results"]:
                        metrics = full_results["results"][subtask]

                        if "acc_norm,none" in metrics:
                            metric_name = "acc_norm,none"
                        elif "acc,none" in metrics:
                            metric_name = "acc,none"
                        elif "word_perplexity,none" in metrics:
                            metric_name = "word_perplexity,none"
                        else:
                            metric_name = None

                        result = {
                            "repeat": repeat,
                            "hostname": socket.gethostname(),
                            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                            "log_path": None,
                            "model": model,
                            "task": subtask,
                            "rht": rht,
                            "metric_name": metric_name,
                            "metric_value": metrics.get(metric_name),
                            "ptq_method": ptq_method.value,
                            "quantization_scheme": (
                                quantization_scheme.value
                                if quantization_scheme is not None
                                else None
                            ),
                            "results": full_results["results"][subtask],
                        }

                        print("RESULT:")
                        print(result)

                        session.add(Experiment(**result))
                        session.commit()
                elif task.task_type == TaskType.reasoning:
                    import inspect_ai

                    eval_kwargs = {}

                    if model == "Qwen/Qwen3.5-35B-A3B":
                        eval_kwargs["extra_body"] = {
                            "temperature": 1.0,
                            "top_p": 0.95,
                            "top_k": 20,
                            "min_p": 0.0,
                            "presence_penalty": 1.5,
                            "repetition_penalty": 1.0,
                            "max_tokens": task.token_limit,
                        }
                    elif model == "Qwen/Qwen3.5-397B-A17B":
                        eval_kwargs["extra_body"] = {
                            "temperature": 0.6,
                            "top_p": 0.95,
                            "top_k": 20,
                            "min_p": 0.0,
                            "presence_penalty": 0.0,
                            "repetition_penalty": 1.0,
                            "max_tokens": task.token_limit,
                        }
                    elif model == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B":
                        eval_kwargs["extra_body"] = {
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "max_tokens": task.token_limit,
                        }
                    elif model == "MiniMaxAI/MiniMax-M2.5":
                        eval_kwargs["extra_body"] = {
                            "temperature": 1.0,
                            "top_p": 0.95,
                            "top_k": 40,
                        }

                    full_results = inspect_ai.eval(
                        tasks=[task.inspect_name],
                        model_base_url=f"{vllm_server_url}/v1",
                        model=f"vllm/{model}",
                        limit=limit,
                        log_dir="inspect_ai_logs",
                        log_realtime=True,
                        display="log",
                        **eval_kwargs,
                    )

                    for log in full_results:
                        metrics = {
                            k: v.value
                            for score in log.results.scores
                            for k, v in score.metrics.items()
                        }

                        metric_name = "accuracy" if "accuracy" in metrics else None

                        result = {
                            "repeat": repeat,
                            "hostname": socket.gethostname(),
                            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                            "log_path": log.location,
                            "model": model,
                            "task": log.eval.task,
                            "rht": rht,
                            "metric_name": metric_name,
                            "metric_value": metrics.get(metric_name),
                            "ptq_method": ptq_method.value,
                            "quantization_scheme": (
                                quantization_scheme.value
                                if quantization_scheme is not None
                                else None
                            ),
                            "results": metrics,
                        }

                        if modal.is_local():
                            session.add(Experiment(**result))
                            session.commit()
                        else:
                            results.append(result)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    finally:
        cleanup()

    return results


@app.function(
    image=ptq_img,
    timeout=24 * 60 * 60,
    volumes={FOUROVERSIX_CACHE_PATH: cache_volume},
)
def run_ptq_coordinator(models: list[str], **kwargs: dict[str, Any]) -> None:
    if modal.is_local():
        for model in models:
            run_ptq.local(model=model, **kwargs)
    else:
        for model in models:
            run_ptq.remote(model=model, **kwargs)


@click.command()
@click.option("--detach", is_flag=True)
@click.option("--enforce-eager", is_flag=True)
@click.option("--ignore-existing-results", is_flag=True)
@click.option("--limit", type=int)
@click.option("--modal", is_flag=True)
@click.option("--model", "-m", type=str, multiple=True)
@click.option("--num-concurrent", type=int, default=16)
@click.option("--num-repeats", type=int, default=None)
@click.option("--port", type=int, default=8000)
@click.option("--ptq-method", "-p", type=PTQMethod, default=PTQMethod.rtn)
@click.option("--quantization-scheme", "-q", type=QuantizationScheme)
@click.option("--rht", is_flag=True)
@click.option("--tasks", "-t", type=Task, multiple=True)
@click.option("--vllm-server-url", type=str)
def cli(**kwargs: dict[str, Any]) -> None:
    detach = kwargs.pop("detach")
    models = kwargs.pop("model")

    if len(models) == 0:
        msg = "At least one model must be provided"
        raise ValueError(msg)

    if kwargs.pop("modal"):
        with modal.enable_output(), app.run(detach=detach):
            run_ptq_coordinator.remote(models=models, **kwargs)
    else:
        run_ptq_coordinator.local(models=models, **kwargs)


if __name__ == "__main__":
    cli()
