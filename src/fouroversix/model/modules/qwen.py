import warnings
from typing import Any

import torch
from fouroversix.model.config import ModuleQuantizationConfig
from fouroversix.model.quantize import QuantizedModule
from fouroversix.quantize import dequantize, quantize
from torch import nn

try:
    from transformers.integrations.moe import _default_apply_gate, _grouped_linear
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts
except ImportError:
    warnings.warn(
        "Qwen3_5MoeExperts not found, please update transformers to the latest "
        "version to quantize Qwen3.5 models.",
        stacklevel=2,
    )

    Qwen3_5MoeExperts = None


@QuantizedModule.register(Qwen3_5MoeExperts)
class FourOverSixQwenExperts(nn.Module):
    """
    Drop-in replacement for the Qwen3_5MoeExperts layer that
    uses FP4 quantization.
    """

    def __init__(
        self,
        module: Qwen3_5MoeExperts,
        config: ModuleQuantizationConfig,
    ) -> None:
        """
        Initialize the FourOverSixQwenExperts layer.

        Args:
            module (GptOssMLP): The high-precision module that this quantized layer will
                replace.
            config (ModuleQuantizationConfig): The quantization configuration to use for
                the layer.

        """
        super().__init__()

        self.num_experts = module.num_experts
        self.intermediate_dim = module.intermediate_dim
        self.hidden_dim = module.hidden_dim

        self.down_proj = module.down_proj
        self.gate_up_proj = module.gate_up_proj

        self.device = self.down_proj.device
        self.config = config

        self.act_fn = module.act_fn

        self.has_bias = False
        self.is_transposed = False

        self.register_buffer(
            "down_proj_pseudoquant",
            nn.Parameter(torch.empty_like(self.down_proj), requires_grad=False),
        )
        self.register_buffer(
            "gate_up_proj_pseudoquant",
            nn.Parameter(torch.empty_like(self.gate_up_proj), requires_grad=False),
        )

    @property
    def parameters_to_quantize(self) -> tuple[str, ...]:
        """Return high precision parameters to be quantized and deleted."""
        return ("down_proj", "gate_up_proj")

    def get_element_size(self, parameter_name: str) -> float:
        """Get the size of a single element, in bytes, for a parameter."""

        return {"down_proj_pseudoquant": 2, "gate_up_proj_pseudoquant": 2}.get(
            parameter_name,
            getattr(self, parameter_name).element_size(),
        )

    def get_quantized_parameters(
        self,
        parameter_name: str,
        parameter: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Prepare this layer for post-training quantization by quantizing the weight,
        storing the quantized weight, and deleting the original weight. This should not
        be done if the layer is used for training, as training requires storage of the
        high-precision weight.
        """

        return {
            f"{parameter_name}_pseudoquant": dequantize(
                quantize(
                    parameter.reshape(-1, parameter.shape[-1]),
                    self.config.get_weight_config(),
                ),
            ).reshape_as(parameter),
        }

    def forward(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Quantized Qwen3.5 experts forward pass."""

        device = hidden_states.device
        num_top_k = top_k_index.size(-1)
        num_tokens = hidden_states.size(0)
        hidden_dim = hidden_states.size(-1)

        # Reshape for easier indexing
        # S is the number of selected tokens-experts pairs (S = num_tokens * num_top_k)
        token_idx = (
            torch.arange(num_tokens, device=device)
            .unsqueeze(1)
            .expand(-1, num_top_k)
            .reshape(-1)
        )  # (S,)
        sample_weights = top_k_weights.reshape(-1)  # (S,)
        expert_ids = top_k_index.reshape(-1)  # (S,)

        # Get current hidden states for selected samples
        selected_hidden_states = hidden_states[token_idx]

        # Sort by expert for grouped processing
        perm = torch.argsort(expert_ids)
        inv_perm = torch.argsort(perm)
        expert_ids_g = expert_ids[perm]
        sample_weights_g = sample_weights[perm]
        selected_hidden_states_g = selected_hidden_states[perm]

        # Select expert weights and biases for selected samples
        # NOTE: We keep all experts here and rely on offsets to target the active ones.
        # I have already implemented a version that only passes the active experts, but
        # to do so I had to use torch.unique which breaks the graph capture
        # (data-dependent). Also there were no speedup gains from it in my experiments,
        # even in eager mode.
        selected_gate_up = self.gate_up_proj_pseudoquant
        selected_down = self.down_proj_pseudoquant
        selected_gate_up_bias = (
            self.gate_up_proj_bias[expert_ids_g] if self.has_bias else None
        )
        selected_down_bias = (
            self.down_proj_bias[expert_ids_g] if self.has_bias else None
        )

        # Compute offsets for grouped_mm
        # using histc instead of bincount to avoid cuda graph issues
        # With deterministic algorithms, CPU only supports float input, CUDA only
        # supports int input.
        histc_input = (
            expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
        )
        num_tokens_per_expert = torch.histc(
            histc_input,
            bins=self.num_experts,
            min=0,
            max=self.num_experts - 1,
        )
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

        selected_hidden_states_g = dequantize(
            quantize(
                selected_hidden_states_g,
                self.config.get_activation_config(),
            ),
        )

        # --- Up projection per expert (grouped) ---
        gate_up_out = _grouped_linear(
            selected_hidden_states_g,
            selected_gate_up,
            offs=offsets,
            bias=selected_gate_up_bias,
            is_transposed=self.is_transposed,
        )  # (S, 2 * intermediate_dim)

        # Apply gating
        gated_out = _default_apply_gate(self, gate_up_out)  # (S, intermediate_dim)

        gated_out = dequantize(
            quantize(gated_out, self.config.get_activation_config()),
        )

        # --- Down projection per expert (grouped) ---
        out_per_sample_g = _grouped_linear(
            gated_out,
            selected_down,
            offs=offsets,
            bias=selected_down_bias,
            is_transposed=self.is_transposed,
        )  # (S, hidden_dim)

        # Apply routing weights
        out_per_sample_g = out_per_sample_g * sample_weights_g.unsqueeze(
            -1,
        )  # (S, hidden_dim)

        # Restore original order
        out_per_sample = out_per_sample_g[inv_perm]

        # Accumulate results using deterministic reshape+sum instead of index_add_
        # (index_add_ with duplicate indices is non-deterministic on CUDA due to
        # atomicAdd)
        final_hidden_states = out_per_sample.view(
            num_tokens,
            num_top_k,
            hidden_dim,
        ).sum(dim=1)

        return final_hidden_states.to(hidden_states.dtype)
