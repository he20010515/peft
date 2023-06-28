import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from transformers.pytorch_utils import Conv1D

from ..import_utils import is_bnb_4bit_available, is_bnb_available
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

from .lora import LoraConfig,LoraModel,LoraLayer
# had to adapt it for `lora_only` to work
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
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError




class Prototype(nn.Module):
    #! Current version only support lora
    def __init__(self,r,in_features,out_features) -> None:
        super().__init__()
        self.lora_A_prototype = nn.Linear(in_features,r,bias=False)
        self.lora_B_prototype = nn.Linear(r,out_features,bias=False)

class subnet_puring(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class MaskedPrototype(nn.Module):
    #! Current version only support lora
    def __init__(self,parent:Prototype,sparsity:float) -> None:
        super().__init__()
        self.parent = parent
        self.sparsity = sparsity
        self.trainable_floating_scoreA = nn.Parameter(
            torch.ones_like(parent.lora_A_prototype.weight,dtype=torch.float32)
        )
        self.trainable_floating_scoreB = nn.Parameter(
            torch.ones_like(parent.lora_B_prototype.weight,dtype=torch.float32)
        )
    def forward(self,x):
        A = subnet_puring.apply(self.trainable_floating_scoreA,self.sparsity) * self.parent.lora_A_prototype.weight
        B = subnet_puring.apply(self.trainable_floating_scoreB,self.sparsity) * self.parent.lora_B_prototype.weight
        x = F.linear(x,A)
        x = F.linear(x,B)
        return x
    # TODO: 需要实现此处对应的lora-binary权重落盘函数.       

@dataclass
class PrototypeLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`LoraModel`].

    Args:
        r (`int`): Lora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`int`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the LoRA transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the LoRA
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
    """
    sparsity:Optional[float] = field(
        default=0.1,
        metadata={"help":"the sparsity of the binary mask"}
    )
    def __post_init__(self):
        self.peft_type = PeftType.PROTOTYPE_LORA

class PrototypeLoraLayer:
    def __init__(self, in_features: int, out_features: int,sparsity:float,parent:Prototype, **kwargs,):
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
        self.sparsity = sparsity
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
            self.lora_AB.update(nn.ModuleDict({adapter_name: MaskedPrototype(parent=self.parent,sparsity=self.sparsity)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)
   
    def reset_lora_parameters(self, adapter_name):
        warnings.warn(" reset_lora_parameters NOT Implemented ,current use One's init")
class Linear(nn.Linear,PrototypeLoraLayer):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        sparsity: float,
        prototype_pointer:Prototype,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ) -> None:
        nn.Linear.__init__(self,in_features, out_features,bias=False)
        PrototypeLoraLayer.__init__(self,in_features,out_features,sparsity=sparsity,parent=prototype_pointer)
        # Freezing the pre-trained weight matrix
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.weight.requires_grad = False
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name
    def merge(self):
        raise NotImplementedError("Not Implemented self.merge")
    def unmerge(self):
        raise NotImplementedError("Not Implemented self.merge")
    def forward(self, x: Tensor) -> Tensor:
        #TODO: support merge method
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_AB.keys():
            return F.linear(x,transpose(self.weight,self.fan_in_fan_out),bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter]>0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter]>0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            
            x = x.to(self.parent.lora_A_prototype.weight.dtype)
            
            result += self.lora_AB["default"](x)
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        result = result.to(previous_dtype)
        return result
    

class PrototypeLoraModel(nn.Module):
    """
    Create a Prototype Lora Model from a pretrained transformers model.

    Args:
        LoraModel ([~transformers.PreTrainedModel]): The model to be adapted
        
    Returns:
        `torch.nn.Module`: The Prototype Lora Model
    """
    # 这里我很想继承LoraModel这个类, 但是我不知道如何在调用父类构造方法之前运行自己的构造方法.
    
    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.prepare_inputs_for_generation = model.prepare_inputs_for_generation
        self._prepare_encoder_decoder_kwargs_for_generation = model._prepare_encoder_decoder_kwargs_for_generation
        lora_input = model.config.d_model
        lora_output = model.config.d_model
        self.prototype = Prototype(r=config['default'].r,in_features=lora_input,out_features=lora_output)
        self.add_adapter(adapter_name,self.peft_config[adapter_name])
        pass
    
    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if not self._check_target_module_exists(lora_config,key):
                continue
            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)
            if isinstance(target,PrototypeLoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            else:
                new_module = self._create_new_module(lora_config,adapter_name,target)
                self._replace_module(parent,target_name,new_module,target)
                
    def _create_new_module(self, lora_config, adapter_name, target):
        bias = hasattr(target,"bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        if isinstance(target,torch.nn.Linear):
            in_features, out_features = target.in_features, target.out_features
            if kwargs['fan_in_fan_out']:
                warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        else:
            raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Cuttently, only `torch.nn.Linear` is supported"
                )
        new_module = Linear(adapter_name,in_features,out_features,sparsity=lora_config.sparsity,bias=bias,prototype_pointer = self.prototype,**kwargs)
        return new_module
                
    def add_adapter(self,adapter_name, config=None):
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
        
    def __getattr__(self, name: str) -> Tensor | Module:
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)