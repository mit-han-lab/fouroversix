from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import modal

from ..resources import FOUROVERSIX_CACHE_PATH, app, cache_volume, hf_secret
from .rtn import RTNEvaluatorImpl, rtn_img

if TYPE_CHECKING:
    from pathlib import Path

    from fouroversix.utils import AdaptiveBlockScalingRule, DataType


with rtn_img.imports():
    import torch
    from fouroversix import fp4_matmul, quantize_model
    from fouroversix.model import FP4Linear
    from transformers import AutoModelForCausalLM


class FP4LinearWithSmoothing(FP4Linear):
    """Drop-in replacement for `FP4Linear` that implements SmoothQuant-style scaling."""

    def __init__(
        self,
        *args: list[Any],
        smoothquant_alpha: float,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(*args, **kwargs)
        self.smoothquant_alpha = smoothquant_alpha

    def apply_ptq(self) -> None:
        """
        Override the parent method to do nothing, since we need the high-precision
        weight when doing PTQ with SmoothQuant.
        """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass for the FP4 linear layer with SmoothQuant-style scaling."""

        out = torch.empty(
            *input.shape[:-1],
            self.weight.shape[0],
            device=input.device,
            dtype=self.out_dtype,
        )

        # Slow bmm
        for i in range(input.shape[0]):
            s = (input[i].abs().max(dim=0).values ** self.smoothquant_alpha) / (
                self.weight.abs().max(dim=0).values ** (1 - self.smoothquant_alpha)
            )

            out[i] = fp4_matmul(
                input[i] / s[None, :],
                self.weight * s[None, :],
                fp4_format=self.fp4_format,
                out_dtype=self.out_dtype,
                out_shape=(input.shape[1], self.weight.shape[0]),
                a_quantize_kwargs={
                    "scale_rule": self.a_scale_rule,
                    "fp4_format": self.fp4_format,
                    **(self.a_quantize_kwargs or {}),
                },
                b_quantize_kwargs={
                    "scale_rule": self.w_scale_rule,
                    "fp4_format": self.fp4_format,
                    **(self.w_quantize_kwargs or {}),
                },
            )

        if self.bias is not None:
            out = out + self.bias

        return out


@app.cls(
    image=rtn_img,
    gpu="B200",
    secrets=[hf_secret],
    timeout=24 * 60 * 60,
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
)
class SmoothQuantEvaluator(RTNEvaluatorImpl):
    """Evaluate a model using SmoothQuant."""

    def quantize_model(
        self,
        model_name: str,
        *,
        device: str,
        dtype: DataType,
        model_kwargs: dict[str, Any] | None = None,
        smoothquant_alpha: float,
        **kwargs: dict[str, Any],
    ) -> AutoModelForCausalLM:
        """Quantize a model using SmoothQuant."""

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            dtype=dtype.torch(),
            **(model_kwargs or {}),
        )

        quantize_model(
            model,
            linear_cls=FP4LinearWithSmoothing,
            smoothquant_alpha=smoothquant_alpha,
            **kwargs,
        )

        return model

    @modal.method()
    def smoothquant_evaluate(
        self,
        model_name: str,
        smoothquant_alpha: float,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a model using SmoothQuant."""

        return super().evaluate_impl(
            model_name,
            smoothquant_alpha=smoothquant_alpha,
            **kwargs,
        )


@app.cls(
    image=rtn_img,
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
    timeout=24 * 60 * 60,
    nonpreemptible=True,
)
class SmoothQuantAutoAlphaEvaluator:
    """Evaluate a model using SmoothQuant."""

    @modal.method()
    def evaluate(
        self,
        model_name: str,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Quantize a model using SmoothQuant."""

        a_scale_rule = kwargs.get("a_scale_rule")
        w_scale_rule = kwargs.get("w_scale_rule")

        smoothquant_alpha = get_smoothquant_alpha(
            model_name,
            a_scale_rule,
            w_scale_rule,
        )

        if smoothquant_alpha is None:
            alpha_candidates = [x / 10 for x in range(11)]

            best_ppl = None

            for i, result in enumerate(
                SmoothQuantEvaluator().smoothquant_evaluate.starmap(
                    [(model_name, alpha) for alpha in alpha_candidates],
                    kwargs={
                        **kwargs,
                        "tasks": ["wikitext_train"],
                    },
                ),
            ):
                ppl = result["results"]["wikitext_train"]["word_perplexity,none"]

                if smoothquant_alpha is None or ppl < best_ppl:
                    smoothquant_alpha = alpha_candidates[i]
                    best_ppl = ppl

                print(f"alpha={alpha_candidates[i]}, ppl={ppl}")

            save_path = get_smoothquant_save_path(
                model_name,
                a_scale_rule,
                w_scale_rule,
            )

            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w") as f:
                f.write(str(smoothquant_alpha))

        return SmoothQuantEvaluator().smoothquant_evaluate.remote(
            model_name,
            smoothquant_alpha=smoothquant_alpha,
            **kwargs,
        )


def get_smoothquant_save_path(
    model_name: str,
    a_scale_rule: AdaptiveBlockScalingRule,
    w_scale_rule: AdaptiveBlockScalingRule,
) -> Path:
    return (
        FOUROVERSIX_CACHE_PATH
        / "ptq"
        / "smoothquant"
        / f"{model_name}-{a_scale_rule.value}-{w_scale_rule.value}"
    )


def get_smoothquant_alpha(
    model_name: str,
    a_scale_rule: AdaptiveBlockScalingRule,
    w_scale_rule: AdaptiveBlockScalingRule,
) -> float | None:
    save_path = get_smoothquant_save_path(model_name, a_scale_rule, w_scale_rule)
    smoothquant_alpha = None

    if save_path.exists():
        with save_path.open("r") as f, contextlib.suppress(ValueError):
            smoothquant_alpha = float(f.read())

    return smoothquant_alpha
