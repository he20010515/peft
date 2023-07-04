import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F
import warnings
from ..utils import (
    COMMON_LAYERS_PATTERN,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple, Union
import re


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, DiscreteKVLoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


@dataclass
class GlobalMemLoraConfig(PeftConfig):
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    init_lora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the Lora layers."},
    )
    codebook_input_channel: int = field(default=16)
    codebook_num: int = field(default=64)
    kv_pairs_num: int = field(default=64)
    target_modules: Optional[list] = None

    def __post_init__(self):
        self.peft_type = PeftType.DISCRETEKV_LORA


class _CodeBook(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, kvpairs_num: int = 8) -> None:
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.embedding = nn.Embedding(
            kvpairs_num,
            input_channel,
        )
        nn.init.uniform_(self.embedding.weight, -1.5, 1.5)  # 初始化键的权重
        self.embedding.requires_grad_(False)
        self.values = nn.Parameter(torch.randn(kvpairs_num, output_channel))

    def forward(self, x: torch.Tensor):
        batchsz, lenseq, _ = x.shape
        view_x = x.view(batchsz * lenseq, 1, self.input_channel)
        view_key = self.embedding.weight.unsqueeze(0)
        distances = torch.cdist(view_x, view_key).squeeze()  # 欧氏距离计算
        min_index = distances.argmin(dim=-1)
        # print(min_index.unique().to("cpu").numpy())
        y = F.embedding(min_index, self.values)
        y = y.view(batchsz, lenseq, self.output_channel)
        return y


class _DiscreteKV(nn.Module):
    def __init__(
        self,
        input_channel: int,  # size of input feature
        codebook_input_channel: int,  #
        output_channel: int,  #
        codebook_num: int,
        kv_pairs_num: int,
    ) -> None:
        super().__init__()
        assert output_channel % codebook_num == 0
        self.codebooks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_channel, codebook_input_channel, bias=False),
                    _CodeBook(codebook_input_channel, output_channel // codebook_num, kv_pairs_num),
                )
                for _ in range(codebook_num)
            ]
        )
        for l in self.codebooks:
            l[0].weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ys = [m(x) for m in self.codebooks]
        ys = torch.concat(ys, dim=-1)  # ys.shape [batch_size,output_channels]
        return ys




class Global_Memory_KV_Lora(nn.Module):
    def __init__(self, model_dim, r, config: GlobalMemLoraConfig) -> None:
        """
        model_dim : 模型维度
        r: lora超参
        """
        self.model_dim = model_dim
        self.r = r
        super().__init__()
        self.lora_A = _DiscreteKV(
            input_channel=model_dim,
            codebook_input_channel=config.codebook_input_channel,
            codebook_num=config.codebook_num,
            output_channel=model_dim * r,
            kv_pairs_num=config.kv_pairs_num,
        )
        self.lora_B = _DiscreteKV(
            input_channel=model_dim,
            codebook_input_channel=config.codebook_input_channel,
            codebook_num=config.codebook_num,
            output_channel=model_dim * r,
            kv_pairs_num=config.kv_pairs_num,
        )

    def forward(self, x):
        # x: [b, n, d]
        b, n, r, d = *x.shape[:2], self.r, self.model_dim
        A = self.lora_A(x).view([b, n, r, d])  # [b n (r*d)] -> [b n r d]
        B = self.lora_B(x).view([b, n, r, d])  # [b n (r*d)] -> [b n r d]
        x = torch.einsum("b n d, b n r d -> b n r", [x, A])
        x = torch.einsum("b n r, b n r d -> b n d", [x, B])
        return x


class Local_Memory_KV_lora(nn.Module):
    def __init__(self, parent: Global_Memory_KV_Lora) -> None:
        super().__init__()
        self.parent = parent

    def forward(self, x):
        x = self.parent(x)
        return x


class DiscreteKVLoraLayer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        parent: Global_Memory_KV_Lora,
        **kwargs,
    ):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_AB = nn.ModuleDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs
        self.parent = parent

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_AB.update(nn.ModuleDict({adapter_name: Local_Memory_KV_lora(parent=self.parent)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        warnings.warn("reset_lora_parameters NOT Implemented ,current use One's init")


class Linear(nn.Linear, DiscreteKVLoraLayer):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        global_mem_pointer: Global_Memory_KV_Lora,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ) -> None:
        nn.Linear.__init__(self, in_features, out_features, bias=False)
        DiscreteKVLoraLayer.__init__(self, in_features, out_features, parent=global_mem_pointer)
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.weight.requires_grad = False
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def forward(self, x: Tensor) -> Tensor:
        # TODO: support merge method
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_AB.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            x = x.to(previous_dtype)

            result += self.lora_AB["default"](x)
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        result = result.to(previous_dtype)
        return result


class GlobalMemLoraModel(nn.Module):
    def __init__(
        self,
        model,
        config: GlobalMemLoraConfig,
        adapter_name,
    ) -> None:
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.prepare_inputs_for_generation = model.prepare_inputs_for_generation
        self._prepare_encoder_decoder_kwargs_for_generation = model._prepare_encoder_decoder_kwargs_for_generation
        try:
            lora_input = model.config.d_model
            lora_output = model.config.d_model
        except:
            lora_input = model.config.hidden_size
            lora_output = model.config.hidden_size
        self.global_mem = Global_Memory_KV_Lora(r=config["default"].r, model_dim=lora_input, config=config["default"])
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_lora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "LoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    @staticmethod
    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config

    def _check_target_module_exists(self, lora_config, key):
        if isinstance(lora_config.target_modules, str):
            target_module_found = re.fullmatch(lora_config.target_modules, key)
        else:
            target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            is_using_layer_indexes = getattr(lora_config, "layers_to_transform", None) is not None
            layer_indexing_pattern = getattr(lora_config, "layers_pattern", None)

            if is_using_layer_indexes and target_module_found:
                layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

                for pattern in layers_pattern:
                    layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                    if layer_index is not None:
                        layer_index = int(layer_index.group(1))
                        if isinstance(lora_config.layers_to_transform, int):
                            target_module_found = layer_index == lora_config.layers_to_transform
                        else:
                            target_module_found = layer_index in lora_config.layers_to_transform

                        break
                    else:
                        target_module_found = False
        return target_module_found

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)
            if "ranknum" in name:
                module.to(old_module.weight.device)

    def _create_new_module(self, lora_config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
        }
        if isinstance(target, torch.nn.Linear):
            in_features, out_features = target.in_features, target.out_features
        else:
            raise ValueError(
                f"Target module {target} is not supported. " f"Cuttently, only `torch.nn.Linear` is supported"
            )
        new_module = Linear(adapter_name, in_features, out_features, global_mem_pointer=self.global_mem, **kwargs)
        return new_module

    def __getattr__(self, name: str) -> Tensor | Module:
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue
            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)
            if isinstance(target, DiscreteKVLoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            else:
                new_module = self._create_new_module(lora_config, adapter_name, target)
                self._replace_module(parent, target_name, new_module, target)

    pass
