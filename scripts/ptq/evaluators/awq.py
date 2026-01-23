from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...resources import (
    FOUROVERSIX_CACHE_PATH,
    Dependency,
    app,
    cache_volume,
    get_image,
    hf_secret,
)
from .rtn import RTNEvaluatorImpl

awq_img = get_image(dependencies=[Dependency.fouroversix, Dependency.awq])

with awq_img.imports():
    import torch
    from awq.quantize.pre_quant import apply_awq, run_awq
    from fouroversix import quantize_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from fouroversix.utils import AdaptiveBlockScalingRule, DataType


@app.cls(
    image=awq_img,
    gpu="B200",
    secrets=[hf_secret],
    timeout=24 * 60 * 60,
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
)
class AWQEvaluator(RTNEvaluatorImpl):
    """Evaluate a model using AWQ."""

    def quantize_model(
        self,
        model_name: str,
        *,
        device: str,
        dtype: DataType,
        a_scale_rule: AdaptiveBlockScalingRule,
        w_scale_rule: AdaptiveBlockScalingRule,
        model_kwargs: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ) -> AutoModelForCausalLM:
        """Quantize a model using AWQ."""

        save_path = (
            FOUROVERSIX_CACHE_PATH
            / "ptq"
            / "awq"
            / f"{model_name}-{a_scale_rule.value}-{w_scale_rule.value}"
        )

        if not save_path.exists():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                dtype=dtype.torch(),
                **(model_kwargs or {}),
            ).eval()

            enc = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                trust_remote_code=True,
            )

            awq_results = run_awq(
                model,
                enc,
                w_bit=4,
                a_bit=4,
                w_q_config={
                    "zero_point": False,
                    "q_group_size": -1,
                    "scale_rule": w_scale_rule,
                },
                a_q_config={
                    "zero_point": False,
                    "q_group_size": -1,
                    "scale_rule": a_scale_rule,
                },
                n_samples=128,
                seqlen=512,
                calib_data="wikitext",
            )

            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(awq_results, save_path)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            dtype=dtype.torch(),
            **(model_kwargs or {}),
        )

        # Apply AWQ
        awq_results = torch.load(save_path, map_location="cpu")
        apply_awq(model, awq_results)

        # Quantize the model
        quantize_model(
            model,
            a_scale_rule=a_scale_rule,
            w_scale_rule=w_scale_rule,
            **kwargs,
        )

        return model.to(device)
