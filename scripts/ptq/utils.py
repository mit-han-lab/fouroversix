from enum import Enum


class PTQMethod(Enum):
    """A PTQ method."""

    rtn = "rtn"
    smoothquant = "smoothquant"
    awq = "awq"


class QuantizationScheme(Enum):
    """A quantization scheme."""

    fouroversix = "fouroversix"
    nvfp4 = "nvfp4"
    mxfp4 = "mxfp4"
    mxfp6_e2m3 = "mxfp6_e2m3"
    mxfp6_e3m2 = "mxfp6_e3m2"
    nvint4 = "nvint4"
    nvint6 = "nvint6"
    if4 = "if4"
    nvfp6_e2m3 = "nvfp6_e2m3"
    nvfp6_e3m2 = "nvfp6_e3m2"
    if6_e2m3 = "if6_e2m3"
    if6_e3m2 = "if6_e3m2"
    mxfp3 = "mxfp3"
    mxfp3_bs8 = "mxfp3_bs8"
    nvfp3 = "nvfp3"
    nvfp3_bs8 = "nvfp3_bs8"
    nvint3 = "nvint3"
    nvint3_bs8 = "nvint3_bs8"
    if3 = "if3"
    if3_bs8 = "if3_bs8"
    mxfp4_bs8 = "mxfp4_bs8"
    nvfp4_bs8 = "nvfp4_bs8"
    fouroversix_bs8 = "fouroversix_bs8"
    nvint4_bs8 = "nvint4_bs8"
    if4_bs8 = "if4_bs8"

    @property
    def dtype(self) -> str:
        return {
            "mxfp4": "mxfp4",
            "nvint4": "nvint4",
            "nvfp4": "nvfp4",
            "fouroversix": "nvfp4",
            "if4": "if4",
        }[self.value]

    @property
    def scale_rule(self) -> str:
        return {
            "mxfp4": "static_6",
            "nvint4": "static_6",
            "nvfp4": "static_6",
            "fouroversix": "mse",
            "if4": "mse",
        }[self.value]

    @property
    def vllm_quantization_name(self) -> str:
        """Get the name of a quantization scheme when using vLLM."""
        return {
            "fouroversix": "fouroversix",
            "nvfp4": "fouroversix_nvfp4",
            "mxfp4": "fouroversix_mxfp4",
            "mxfp6_e2m3": "fouroversix_mxfp6_e2m3",
            "mxfp6_e3m2": "fouroversix_mxfp6_e3m2",
            "nvint4": "fouroversix_nvint4",
            "nvint6": "fouroversix_nvint6",
            "if4": "fouroversix_if4",
            "nvfp6_e2m3": "fouroversix_nvfp6_e2m3",
            "nvfp6_e3m2": "fouroversix_nvfp6_e3m2",
            "if6_e2m3": "fouroversix_if6_e2m3",
            "if6_e3m2": "fouroversix_if6_e3m2",
            "mxfp3": "fouroversix_mxfp3",
            "mxfp3_bs8": "fouroversix_mxfp3_bs8",
            "nvfp3": "fouroversix_nvfp3",
            "nvfp3_bs8": "fouroversix_nvfp3_bs8",
            "nvint3": "fouroversix_nvint3",
            "nvint3_bs8": "fouroversix_nvint3_bs8",
            "if3": "fouroversix_if3",
            "if3_bs8": "fouroversix_if3_bs8",
            "mxfp4_bs8": "fouroversix_mxfp4_bs8",
            "nvfp4_bs8": "fouroversix_nvfp4_bs8",
            "fouroversix_bs8": "fouroversix_bs8",
            "nvint4_bs8": "fouroversix_nvint4_bs8",
            "if4_bs8": "fouroversix_if4_bs8",
        }.get(self.value)


class TaskType(Enum):
    """A type of task."""

    question_answering = "question_answering"
    reasoning = "reasoning"
    perplexity = "perplexity"


class Task(Enum):
    """A task to evaluate."""

    boolq = "boolq"
    arc_easy = "arc_easy"
    arc_challenge = "arc_challenge"
    hellaswag = "hellaswag"
    lambada = "lambada"
    piqa = "piqa"
    aime24 = "aime24"
    aime25 = "aime25"
    gpqa_diamond = "gpqa_diamond"
    mmlu_pro = "mmlu_pro"
    wikitext = "wikitext"
    c4 = "c4"

    @property
    def inspect_name(self) -> str:
        """Get the name of a task when using inspect_ai."""
        return {
            "gpqa_diamond": "inspect_evals/gpqa_diamond",
            "mmlu_pro": "inspect_evals/mmlu_pro",
            "aime24": "inspect_evals/aime2024",
            "aime25": "inspect_evals/aime2025",
        }.get(self.value)

    @property
    def task_type(self) -> TaskType:
        """Get the type of a task."""
        return {
            "boolq": TaskType.question_answering,
            "arc_easy": TaskType.question_answering,
            "arc_challenge": TaskType.question_answering,
            "hellaswag": TaskType.question_answering,
            "lambada": TaskType.question_answering,
            "piqa": TaskType.question_answering,
            "aime24": TaskType.reasoning,
            "aime25": TaskType.reasoning,
            "gpqa_diamond": TaskType.reasoning,
            "mmlu_pro": TaskType.reasoning,
            "wikitext": TaskType.perplexity,
            "c4": TaskType.perplexity,
        }.get(self.value)

    @property
    def token_limit(self) -> int | None:
        """Get the token limit for a task."""
        return {
            "aime24": 32768,
            "aime25": 32768,
            "gpqa_diamond": 8192,
            "mmlu_pro": 4096,
        }.get(self.value)

    @property
    def num_repeats(self) -> int:
        """Get the number of times a task should be repeated."""
        return {
            "boolq": 3,
            "arc_easy": 3,
            "arc_challenge": 3,
            "lambada": 3,
            "piqa": 3,
        }.get(self.value, 1)
