import torch
import torch.nn as nn
from torchao.float8.float8_utils import (
    amax_to_scale,
    tensor_to_amax,
    to_fp8_saturated,
)
from torch.nn import init
import math
from torch.compiler import is_compiling


try:
    from cublas_ops import CublasLinear
except ImportError:
    CublasLinear = type(None)


class F8Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        float8_dtype=torch.float8_e4m3fn,
        float_weight: torch.Tensor = None,
        float_bias: torch.Tensor = None,
        num_scale_trials: int = 24,
        input_float8_dtype=torch.float8_e5m2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.float8_dtype = float8_dtype
        self.input_float8_dtype = input_float8_dtype
        self.input_scale_initialized = False
        self.weight_initialized = False
        self.max_value = torch.finfo(self.float8_dtype).max
        self.input_max_value = torch.finfo(self.input_float8_dtype).max
        factory_kwargs = {"dtype": dtype, "device": device}
        if float_weight is None:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features), **factory_kwargs)
            )
        else:
            self.weight = nn.Parameter(
                float_weight, requires_grad=float_weight.requires_grad
            )
        if float_bias is None:
            if bias:
                self.bias = nn.Parameter(
                    torch.empty(out_features, **factory_kwargs),
                    requires_grad=bias.requires_grad,
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(float_bias, requires_grad=float_bias.requires_grad)
        self.num_scale_trials = num_scale_trials
        self.input_amax_trials = torch.zeros(
            num_scale_trials, requires_grad=False, device=device, dtype=torch.float32
        )
        self.trial_index = 0
        self.register_buffer("scale", None)
        self.register_buffer(
            "input_scale",
            None,
        )
        self.register_buffer(
            "float8_data",
            None,
        )
        self.scale_reciprocal = self.register_buffer("scale_reciprocal", None)
        self.input_scale_reciprocal = self.register_buffer(
            "input_scale_reciprocal", None
        )

    def quantize_weight(self):
        if self.weight_initialized:
            return
        amax = tensor_to_amax(self.weight.data)
        scale = amax_to_scale(amax, self.float8_dtype, self.weight.dtype)
        self.float8_data = to_fp8_saturated(self.weight.data * scale, self.float8_dtype)
        self.scale = scale.float()
        self.weight_initialized = True
        self.scale_reciprocal = self.scale.reciprocal().float()
        self.weight.data = torch.zeros(
            1, dtype=self.weight.dtype, device=self.weight.device, requires_grad=False
        )

    def quantize_input(self, x: torch.Tensor):
        if self.input_scale_initialized:
            return to_fp8_saturated(x * self.input_scale, self.input_float8_dtype)
        elif self.trial_index < self.num_scale_trials:
            amax = tensor_to_amax(x)
            self.input_amax_trials[self.trial_index] = amax
            self.trial_index += 1
            self.input_scale = amax_to_scale(
                self.input_amax_trials[: self.trial_index].max(),
                self.input_float8_dtype,
                self.weight.dtype,
            )
            self.input_scale_reciprocal = self.input_scale.reciprocal()
            return to_fp8_saturated(x * self.input_scale, self.input_float8_dtype)
        else:
            self.input_scale = amax_to_scale(
                self.input_amax_trials.max(), self.input_float8_dtype, self.weight.dtype
            )
            self.input_scale_reciprocal = self.input_scale.reciprocal()
            self.input_scale_initialized = True
            return to_fp8_saturated(x * self.input_scale, self.input_float8_dtype)

    def reset_parameters(self) -> None:
        if self.weight_initialized:
            self.weight = nn.Parameter(
                torch.empty(
                    (self.out_features, self.in_features),
                    **{
                        "dtype": self.weight.dtype,
                        "device": self.weight.device,
                    },
                )
            )
            self.weight_initialized = False
            self.input_scale_initialized = False
            self.trial_index = 0
            self.input_amax_trials.zero_()
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        self.quantize_weight()
        self.max_value = torch.finfo(self.float8_dtype).max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_scale_initialized or is_compiling():
            x = (
                x.mul(self.input_scale)
                .clamp(min=-self.input_max_value, max=self.input_max_value)
                .type(self.input_float8_dtype)
            )
        else:
            x = self.quantize_input(x)

        prev_dims = x.shape[:-1]

        x = x.view(-1, self.in_features)

        # float8 matmul, much faster than float16 matmul w/ float32 accumulate on ADA devices!
        return torch._scaled_mm(
            x,
            self.float8_data.T,
            self.input_scale_reciprocal,
            self.scale_reciprocal,
            bias=self.bias,
            out_dtype=self.weight.dtype,
            use_fast_accum=True,
        ).view(*prev_dims, self.out_features)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        float8_dtype=torch.float8_e4m3fn,
        input_float8_dtype=torch.float8_e5m2,
    ):
        f8_lin = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            float8_dtype=float8_dtype,
            float_weight=linear.weight.data,
            float_bias=(linear.bias.data if linear.bias is not None else None),
            input_float8_dtype=input_float8_dtype,
        )
        f8_lin.quantize_weight()
        return f8_lin


def recursive_swap_linears(
    model: nn.Module,
    float8_dtype=torch.float8_e4m3fn,
    input_float8_dtype=torch.float8_e5m2,
):
    """
    Recursively swaps all nn.Linear modules in the given model with F8Linear modules.

    This function traverses the model's structure and replaces each nn.Linear
    instance with an F8Linear instance, which uses 8-bit floating point
    quantization for weights. The original linear layer's weights are deleted
    after conversion to save memory.

    Args:
        model (nn.Module): The PyTorch model to modify.

    Note:
        This function modifies the model in-place. After calling this function,
        all linear layers in the model will be using 8-bit quantization.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and not isinstance(
            child, (F8Linear, CublasLinear)
        ):

            setattr(
                model,
                name,
                F8Linear.from_linear(
                    child,
                    float8_dtype=float8_dtype,
                    input_float8_dtype=input_float8_dtype,
                ),
            )
            del child
        else:
            recursive_swap_linears(child)


@torch.inference_mode()
def quantize_flow_transformer_and_dispatch_float8(
    flow_model: nn.Module,
    device=torch.device("cuda"),
    float8_dtype=torch.float8_e4m3fn,
    input_float8_dtype=torch.float8_e5m2,
    offload_flow=False,
):
    """
    Quantize the flux flow transformer model (original BFL codebase version) and dispatch to the given device.
    """
    for i, module in enumerate(flow_model.double_blocks):
        module.to(device)
        module.eval()
        recursive_swap_linears(
            module, float8_dtype=float8_dtype, input_float8_dtype=input_float8_dtype
        )
        torch.cuda.empty_cache()
    for i, module in enumerate(flow_model.single_blocks):
        module.to(device)
        module.eval()
        recursive_swap_linears(
            module, float8_dtype=float8_dtype, input_float8_dtype=input_float8_dtype
        )
        torch.cuda.empty_cache()
    to_gpu_extras = [
        "vector_in",
        "img_in",
        "txt_in",
        "time_in",
        "guidance_in",
        "final_layer",
        "pe_embedder",
    ]
    for module in to_gpu_extras:
        m_extra = getattr(flow_model, module)
        if m_extra is None:
            continue
        m_extra.to(device)
        m_extra.eval()
        if isinstance(m_extra, nn.Linear) and not isinstance(
            m_extra, (F8Linear, CublasLinear)
        ):
            setattr(
                flow_model,
                module,
                F8Linear.from_linear(
                    m_extra,
                    float8_dtype=float8_dtype,
                    input_float8_dtype=input_float8_dtype,
                ),
            )
            del m_extra
        elif module != "final_layer":
            recursive_swap_linears(
                m_extra,
                float8_dtype=float8_dtype,
                input_float8_dtype=input_float8_dtype,
            )
        torch.cuda.empty_cache()
    if offload_flow:
        flow_model.to("cpu")
        torch.cuda.empty_cache()
    return flow_model
