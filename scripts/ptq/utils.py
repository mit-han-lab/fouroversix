from __future__ import annotations

from enum import Enum
from typing import Any


class PTQMethod(str, Enum):
    """Methods of post-training quantization."""

    awq = "awq"
    high_precision = "high_precision"
    gptq = "gptq"
    rtn = "rtn"
    smoothquant = "smoothquant"
    spinquant = "spinquant"


def get_model_size(model_name: str | None) -> float:
    return float(model_name.split("-")[-1][:-1]) if model_name else 0.0


def print_results(results: dict[str, Any]) -> None:
    if "results" in results:
        print_results(results["results"])
        return

    for task, result in results.items():
        if "acc_norm,none" in result or "acc,none" in result:
            acc = result.get("acc_norm,none", result.get("acc,none", None)) * 100
            print(task, f"{acc:.1f}")
        elif "word_perplexity,none" in result:
            print(f"{task}: {result['word_perplexity,none']:.2f}")
        else:
            print(f"{task}: {result}")
