import torch
from torch import nn as nn
import inspect
from torch.nn import ModuleList
from torch.nn import functional as F

import algorithms.cw_offline.util as util

class FullyConnectedQFunction(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_layers: list = [256, 256, 256],
        act_func_hidden: str = "relu",
        act_func_last: str = None,
        init_method: str = "orthogonal",
        out_layer_gain: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        name: str = "q_func",
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.net = util.MLP(
            name=name,
            dim_in=observation_dim + action_dim,
            dim_out=1,
            hidden_layers=hidden_layers,
            act_func_hidden=act_func_hidden,
            act_func_last=act_func_last,
            init_method=init_method,
            out_layer_gain=out_layer_gain,
            dtype=dtype,
            device=device,
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = self.extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])

        input_tensor = torch.cat([observations, actions], dim=-1)
        q_values = self.net(input_tensor).squeeze(-1)

        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values

    def save(self, log_dir: str, epoch: int):
        self.net.save(log_dir, epoch)

    def load(self, log_dir: str, epoch: int):
        self.net.load(log_dir, epoch)

    def configure_optimizer(self, weight_decay, learning_rate, betas, device_type="cpu"):
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas)

    def extend_and_repeat(self, tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
        return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)