import logging
import numpy as np

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, \
    normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import get_activation_fn, try_import_torch

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class NegotiateModel(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, interaction_hidden_size):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        no_final_linear = model_config.get("no_final_linear")  # TODO Handle no_final_linear
        assert(not no_final_linear, "Not Implemented yet bro")

        self.vf_share_layers = model_config.get("vf_share_layers")
        self.vf_hiddens = model_config.get("vf_hiddens", [10, 10])
        self.free_log_std = model_config.get("free_log_std")
        self.interaction_input_size = 2
        assert(np.product(obs_space.shape) == self.control_input_size + self.interaction_input_size,
               "Wrong size of obs space")
        interaction_hidden_size = interaction_hidden_size
        activation = get_activation_fn(
            model_config.get("fcnet_activation"), framework="torch")

        # Are the std required as output for the action
        self.std = ((num_outputs / 2) == np.product(action_space.shape))

        # Are the log_std varies with state or not
        if self.free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two", num_outputs)
            num_outputs = num_outputs // 2

        self._logits = None  # Output of the network, called logits for consistency with the rest of RLlib

        # Build the Negotiate model
        self.linear_accept_1 = SlimFC(self.interaction_input_size, interaction_hidden_size,
                                      initializer=normc_initializer(1.0),
                                      activation_fn=activation)
        self.linear_accept_2_mean = SlimFC(interaction_hidden_size, 1, initializer=normc_initializer(0.01),
                                           activation_fn=None)
        self.interaction = nn.Sequential(self.linear_accept_1, self.linear_accept_2_mean)
        self.linear_coop_mean = AppendBiasLayer(1)

        if self.std:
            if not self.free_log_std:
                self.linear_accept_2_std = SlimFC(interaction_hidden_size, 1, initializer=normc_initializer(0.01),
                                                  activation_fn=None)
                self.linear_coop_std = AppendBiasLayer(1)
                self.interaction_std = nn.Sequential(self.linear_accept_1, self.linear_accept_2_std)
                self.coop_std = AppendBiasLayer(1)
            else:
                self._append_free_log_std = AppendBiasLayer(num_outputs)

        # value function
        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            prev_vf_layer_size = int(np.product(obs_space.shape))
            vf_layers = []
            for size in self.vf_hiddens:
                vf_layers.append(
                    SlimFC(
                        in_size=prev_vf_layer_size,
                        out_size=size,
                        activation_fn=activation,
                        initializer=normc_initializer(1.0)))
                prev_vf_layer_size = size
                prev_layer_size = prev_vf_layer_size
            self._value_branch_separate = nn.Sequential(*vf_layers)
        else:
            raise NotImplementedError()
        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(1.0),
            activation_fn=None)
        self._value_module = nn.Sequential(self._value_branch_separate, self._value_branch)
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"]
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        interact_in = self._last_flat_in

        # Final layer for the mean
        y_accept_mean = self.interaction(interact_in)
        y = self.linear_coop_mean(y_accept_mean)

        if self.std:
            # Final layer for the std
            if not self.free_log_std:
                y_accept_std = self.interaction(interact_in)
                y_std = self.coop_std(y_accept_std, axis=1))
                return torch.cat((y, y_std), axis=1), state
            else:
                return self._append_free_log_std(y), state
        else:
            return y, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._last_flat_in is not None, "must call forward() first"
        if self._value_branch_separate:
            return self._value_module(self._last_flat_in).squeeze(1)
        else:
            raise NotImplementedError()
