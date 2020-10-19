import logging
from typing import Dict, List

import numpy as np

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, \
    normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import get_activation_fn, try_import_torch, TensorType

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class InvestorModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.x_value = nn.Parameter(torch.zeros(2), True)
        self.h_value_branch = SlimFC(
            in_size=1,
            out_size=2,
            initializer=normc_initializer(1.0),
            activation_fn=None,
            use_bias=True)
        self.out_value_branch = SlimFC(
            in_size=2,
            out_size=1,
            initializer=normc_initializer(1.0),
            activation_fn=None,
            use_bias=True)
        self.value_branch = nn.Sequential(self.h_value_branch, self.out_value_branch)

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (
            TensorType, List[TensorType]):
        return torch.tanh(self.x_value).view(-1, 2), []

    def value_function(self) -> TensorType:
        return self.value_branch(torch.tanh(self.x_value).view(-1, 2)[:, 0])
