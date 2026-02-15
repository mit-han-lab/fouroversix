import torch
from fouroversix.matmul import fp4_matmul
from fouroversix.model.config import LayerQuantizationConfig
from fouroversix.model.quantize import QuantizedLayer
from fouroversix.quantize import QuantizationConfig, QuantizedTensor, quantize_to_fp4
from fouroversix.utils import DataType
from torch import nn
from transformers import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP, GptOssTopKRouter


@QuantizedLayer.register(GptOssMLP)
class FourOverSixGptOssMLP(nn.Module):
    """Drop-in replacement for MoE layer that uses FP4 quantization."""

    def __init__(
        self,
        module: GptOssMLP,
        config: LayerQuantizationConfig,
    ) -> None:
        super().__init__()

        gpt_oss_config = GptOssConfig(
            num_local_experts=module.experts.num_experts,
            hidden_size=module.experts.hidden_size,
            intermediate_size=module.experts.intermediate_size,
            num_experts_per_token=module.router.top_k,
        )

        self.router = GptOssTopKRouter(gpt_oss_config)
        self.router.weight = module.router.weight
        self.router.bias = module.router.bias

        down_proj = []
        gate_up_proj = []
        for e in range(module.experts.down_proj.storage.data.shape[0]):
            down_tensor = QuantizedTensor(
                values=module.experts.down_proj.storage.data[e].permute(1, 0),
                scale_factors=module.experts.down_proj_precision_config.weight_scale.storage.data[
                    e
                ]
                .permute(1, 0)
                .view(torch.float8_e8m0fnu),
                amax=torch.ones(
                    (module.experts.down_proj.storage.data[e].shape[1]),
                    device=module.experts.down_proj.storage.data.device,
                    dtype=torch.float32,
                ),
                dtype=DataType.mxfp4,
                original_shape=(
                    module.experts.down_proj.storage.data.shape[2],
                    module.experts.down_proj.storage.data.shape[1] * 2,
                ),
                scale_rule=config.get_weight_scale_rule(),
            )
            down_proj.append(down_tensor)

        for e in range(module.experts.gate_up_proj.storage.data.shape[0]):
            gate_up_tensor = QuantizedTensor(
                values=module.experts.gate_up_proj.storage.data[e].permute(1, 0),
                scale_factors=module.experts.gate_up_proj_precision_config.weight_scale.storage.data[
                    e
                ]
                .permute(1, 0)
                .view(torch.float8_e8m0fnu),
                amax=torch.ones(
                    (module.experts.gate_up_proj.storage.data[e].shape[1],),
                    device=module.experts.gate_up_proj.storage.data.device,
                    dtype=torch.float32,
                ),
                dtype=DataType.mxfp4,
                original_shape=(
                    module.experts.gate_up_proj.storage.data.shape[2],
                    module.experts.gate_up_proj.storage.data.shape[1] * 2,
                ),
                scale_rule=config.get_weight_scale_rule(),
            )
            gate_up_proj.append(gate_up_tensor)

        self.experts = FourOverSixGptOssExperts(
            gpt_oss_config,
            down_proj=down_proj,
            down_proj_bias=module.experts.down_proj_bias,
            gate_up_proj=gate_up_proj,
            gate_up_proj_bias=module.experts.gate_up_proj_bias,
            device=module.experts.down_proj_bias.device,
            quantization_config=config,
        )

    def apply_ptq(self) -> None:
        """
        Prepare this layer for post-training quantization by quantizing the weight,
        storing the quantized weight, and deleting the original weight. This should not
        be done if the layer is used for training, as training requires storage of the
        high-precision weight.
        """

        self.experts.apply_ptq()

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the FP4 MLP layer."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        _, router_scores, router_indices = self.router(hidden_states)
        hidden_states = self.experts(hidden_states, router_indices, router_scores)
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states, router_scores


class FourOverSixGptOssExperts(nn.Module):
    """Drop-in replacement for MoE layer that uses FP4 quantization."""

    def __init__(
        self,
        config: GptOssConfig,
        down_proj: list[QuantizedTensor],
        gate_up_proj: list[QuantizedTensor],
        down_proj_bias: torch.Tensor = None,
        gate_up_proj_bias: torch.Tensor = None,
        device: torch.device | None = None,
        quantization_config: LayerQuantizationConfig | None = None,
    ) -> None:
        super().__init__()

        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size

        self.down_proj = down_proj
        self.down_proj_bias = down_proj_bias

        self.gate_up_proj = gate_up_proj
        self.gate_up_proj_bias = gate_up_proj_bias

        self.alpha = 1.702
        self.limit = 7.0

        self.device = device
        self.quantization_config = quantization_config

        self.apply_ptq()

    def apply_ptq(self) -> None:
        """
        Prepare this layer for post-training quantization by quantizing the weight,
        storing the quantized weight, and deleting the original weight. This should not
        be done if the layer is used for training, as training requires storage of the
        high-precision weight.
        """

        # already quantized
        if self.quantization_config.dtype == DataType.mxfp4:
            return

        old_down_proj = self.down_proj
        dequantized_experts = []
        for e in range(len(self.down_proj)):
            dequantized = self.down_proj[e].dequantize()
            dequantized_experts.append(dequantized)
        self.down_proj = dequantized_experts
        # Explicitly delete old FP4Tensor objects to free GPU memory
        del old_down_proj

        quantized_down_proj = self.quantized_weight(self.down_proj)
        # Delete intermediate dequantized tensors before reassignment
        del self.down_proj
        self.down_proj = quantized_down_proj

        # Clear GPU cache after down_proj quantization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        old_gate_up_proj = self.gate_up_proj
        dequantized_experts = []
        for e in range(len(self.gate_up_proj)):
            dequantized = self.gate_up_proj[e].dequantize()
            dequantized_experts.append(dequantized)
        self.gate_up_proj = dequantized_experts
        # Explicitly delete old FP4Tensor objects to free GPU memory
        del old_gate_up_proj

        quantized_gate_up_proj = self.quantized_weight(self.gate_up_proj)
        # Delete intermediate dequantized tensors before reassignment
        del self.gate_up_proj
        self.gate_up_proj = quantized_gate_up_proj

        # Clear GPU cache after gate_up_proj quantization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_indices: torch.Tensor = None,
        routing_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass for the FP4 experts layer."""

        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            -1,
            self.hidden_size,
        )
        next_states = torch.zeros_like(
            hidden_states,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                routing_indices,
                num_classes=self.num_experts,
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_i in expert_hit[:]:
            expert_idx = expert_i[0]
            if expert_idx == self.num_experts:
                continue
            with torch.no_grad():
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            # Gate-up projection
            fprop_activation_config = QuantizationConfig(
                backend=self.quantization_config.quantize_backend,
                dtype=self.quantization_config.dtype,
                scale_rule=self.quantization_config.get_activation_scale_rule(),
            )

            gate_up = fp4_matmul(
                current_state,
                self.gate_up_proj[expert_idx],
                input_config=fprop_activation_config,
                out_dtype=self.quantization_config.output_dtype,
            )
            gate_up += self.gate_up_proj_bias[expert_idx]
            del current_state

            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            del gate_up
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            del gate
            gated_output = (up + 1) * glu
            del up, glu

            # Down projection
            out = fp4_matmul(
                gated_output,
                self.down_proj[expert_idx],
                input_config=fprop_activation_config,
                out_dtype=self.quantization_config.output_dtype,
            )
            del gated_output
            out += self.down_proj_bias[expert_idx]
            weighted_output = out * routing_weights[token_idx, top_k_pos, None]
            del out
            next_states.index_add_(
                0,
                token_idx,
                weighted_output.to(hidden_states.dtype),
            )
            del weighted_output

        del expert_mask, expert_hit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return next_states.view(batch_size, -1, self.hidden_size)

    def quantized_weight(
        self,
        weight: list[torch.Tensor] | list[QuantizedTensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the quantized weights. Handles 3D MoE weights [E, K, N] by expert."""

        quantized_weight = []
        for e in range(len(weight)):
            config = QuantizationConfig(
                backend=self.quantization_config.quantize_backend,
                dtype=self.quantization_config.dtype,
                scale_rule=self.quantization_config.get_weight_scale_rule(),
            )
            quantized_tensor = quantize_to_fp4(weight[e], config)
            quantized_weight.append(quantized_tensor)

        return quantized_weight
