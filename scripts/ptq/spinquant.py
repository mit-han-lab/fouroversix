from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any

import modal

from ..resources import (
    FOUROVERSIX_CACHE_PATH,
    FOUROVERSIX_INSTALL_PATH,
    Dependency,
    app,
    cache_volume,
    get_image,
    hf_secret,
)
from .evaluator import PTQEvaluator

spinquant_img = get_image(
    dependencies=[
        Dependency.fast_hadamard_transform,
        Dependency.fouroversix,
        Dependency.spinquant,
    ],
)

with spinquant_img.imports():
    import sys

    sys.path.append((FOUROVERSIX_INSTALL_PATH / "spinquant").as_posix())

    from eval_utils.main import ptq_model
    from transformers import AutoConfig, AutoModelForCausalLM
    from utils.process_args import process_args_ptq

    from .utils import get_model_size

    if TYPE_CHECKING:
        from fouroversix.utils import AdaptiveBlockScalingRule, DataType

MIN_MODEL_SIZE_FOR_8xB200 = 32

SPINQUANT_ARGS = [
    "--model_max_length",
    "8192",
    "--fp16",
    "False",
    "--bf16",
    "True",
    "--w_bits",
    "4",
    "--a_bits",
    "4",
    "--k_bits",
    "16",
    "--v_bits",
    "16",
]


@app.cls(
    image=spinquant_img,
    timeout=24 * 60 * 60,
    secrets=[hf_secret],
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
)
class SpinQuantOptimizer:
    """Optimize a model with SpinQuant."""

    @modal.method()
    def optimize(
        self,
        model_name: str,
        *,
        a_scale_rule: AdaptiveBlockScalingRule,
        w_scale_rule: AdaptiveBlockScalingRule,
        spinquant_save_path: str,
        spinquant_steps: int,
    ) -> None:
        """Optimize a model with SpinQuant."""

        subprocess.run(  # noqa: S603
            [  # noqa: S607
                "torchrun",
                "--nnodes=1",
                "--nproc_per_node=auto",
                f"{FOUROVERSIX_INSTALL_PATH}/spinquant/optimize_rotation.py",
                "--input_model",
                model_name,
                "--output_dir",
                spinquant_save_path,
                "--output_rotation_path",
                spinquant_save_path,
                "--log_on_each_node",
                "False",
                "--per_device_train_batch_size",
                "1",
                "--logging_steps",
                "1",
                "--learning_rate",
                "1.5",
                "--weight_decay",
                "0.",
                "--lr_scheduler_type",
                "cosine",
                "--gradient_checkpointing",
                "True",
                "--save_safetensors",
                "False",
                "--max_steps",
                str(spinquant_steps),
                "--a_scale_rule",
                a_scale_rule.value,
                "--w_scale_rule",
                w_scale_rule.value,
                *SPINQUANT_ARGS,
            ],
            check=True,
        )

        cache_volume.commit()


@app.cls(
    image=spinquant_img,
    timeout=24 * 60 * 60,
    gpu="B200",
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
)
class SpinQuantEvaluator(PTQEvaluator):
    """Evaluate a quantized model with SpinQuant."""

    def quantize_model(
        self,
        model_name: str,
        *,
        device: str,
        dtype: DataType,
        optimized_rotation_path: str,
        **kwargs: dict[str, Any],
    ) -> AutoModelForCausalLM:
        """Export a quantized model with SpinQuant."""

        a_scale_rule = kwargs.get("a_scale_rule")
        w_scale_rule = kwargs.get("w_scale_rule")

        sys.argv = [
            sys.argv[0],
            "--input_model",
            model_name,
            "--do_train",
            "False",
            "--do_eval",
            "True",
            "--per_device_eval_batch_size",
            "4",
            "--rotate",
            "--optimized_rotation_path",
            optimized_rotation_path,
            "--a_scale_rule",
            a_scale_rule.value,
            "--w_scale_rule",
            w_scale_rule.value,
            *SPINQUANT_ARGS,
        ]

        config = AutoConfig.from_pretrained(model_name)

        # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings,
        # clone lm_head from embed_tokens
        process_word_embeddings = False
        if config.tie_word_embeddings:
            config.tie_word_embeddings = False
            process_word_embeddings = True

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=dtype.torch(),
        )

        if process_word_embeddings:
            model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

        model.to(device)
        model_args, _, ptq_args = process_args_ptq()

        cache_volume.reload()

        model = ptq_model(ptq_args, model, model_args)
        model.to(device)

        return model


@app.cls(
    image=spinquant_img,
    timeout=24 * 60 * 60,
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
    nonpreemptible=True,
)
class SpinQuantEvaluationCoordinator:
    """Coordinate evaluation of a quantized model with SpinQuant."""

    @modal.method()
    def evaluate(
        self,
        model_name: str,
        *,
        spinquant_steps: int = 100,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Train and evaluate a quantized model with SpinQuant."""

        a_scale_rule = kwargs.get("a_scale_rule")
        w_scale_rule = kwargs.get("w_scale_rule")

        spinquant_save_path = (
            FOUROVERSIX_CACHE_PATH
            / "ptq"
            / "spinquant"
            / f"{model_name}-{a_scale_rule.value}-{w_scale_rule.value}"
        )

        if not (spinquant_save_path / "R.bin").exists():
            model_is_large = get_model_size(model_name) >= MIN_MODEL_SIZE_FOR_8xB200

            SpinQuantOptimizer.with_options(
                gpu=f"B200:{8 if model_is_large else 1}",
            )().optimize.remote(
                model_name,
                a_scale_rule=a_scale_rule,
                w_scale_rule=w_scale_rule,
                spinquant_save_path=spinquant_save_path.as_posix(),
                spinquant_steps=spinquant_steps,
            )

        return SpinQuantEvaluator().evaluate.remote(
            model_name=model_name,
            optimized_rotation_path=(spinquant_save_path / "R.bin").as_posix(),
            **kwargs,
        )
