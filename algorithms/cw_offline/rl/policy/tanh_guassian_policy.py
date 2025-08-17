import torch
import torch.nn as nn

from algorithms.cw_offline.util.util_mp import *
from algorithms.cw_offline.rl.policy import AbstractGaussianPolicy

from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform


class TanhGaussianPolicy(AbstractGaussianPolicy):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 mean_net_args: dict,
                 variance_net_args: dict,
                 init_method: str,
                 out_layer_gain: float,
                 act_func_hidden: str,
                 act_func_last: str,
                 max_action: float,
                 log_std_multiplier: float = 1.0,
                 log_std_offset: float = -1.0,
                 orthogonal_init: bool = True,
                 no_tanh: bool = False,
                 dtype: str = "torch.float32",
                 device: str = "cpu",
                 **kwargs):
        super().__init__(dim_in,
                         dim_out,
                         mean_net_args,
                         variance_net_args,
                         init_method,
                         out_layer_gain,
                         act_func_hidden,
                         act_func_last,
                         dtype,
                         device,
                         **kwargs)

        self.max_action = max_action
        self.log_std_multiplier = nn.Parameter(torch.tensor(log_std_multiplier, dtype=torch.float32))
        self.log_std_offset = nn.Parameter(torch.tensor(log_std_offset, dtype=torch.float32))
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh
        self.deterministic = kwargs.get("deterministic", False)

    def policy(self, obs):
        mean = self.mean_net(obs)
        if self.contextual_cov:
            if self.std_only:
                log_std = self.variance_net(obs)
                log_std = self.log_std_multiplier * log_std + self.log_std_offset
            else:
                cov_vector = self.variance_net(obs)
                params_L = self._vector_to_cholesky(cov_vector)
                cov_matrix = params_L @ params_L.transpose(-1, -2)
                log_std = torch.sqrt(torch.diagonal(cov_matrix, dim1=-2, dim2=-1))
                log_std = self.log_std_multiplier * log_std + self.log_std_offset
        else:
            log_std = self.log_std_multiplier * torch.exp(self.variance_net.variable) + self.log_std_offset

        log_std = torch.clamp(log_std, min=-20.0, max=2.0)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        return action_distribution, mean, std

    def sample(self, obs, num_samples: int = 1):
        dist, mean, std = self.policy(obs)

        if self.deterministic:
            action = torch.tanh(mean)
            log_prob = dist.log_prob(action).sum(-1)
        else:
            action = dist.rsample(torch.Size([num_samples]))  # [K, B, dim_out]
            log_prob = dist.log_prob(action).sum(-1)  # [K, B]

            # move K to be the second dim: [B, K, ...]
            if action.ndim == 3:
                action = action.transpose(0, 1)  # [B, K, dim_out]
                log_prob = log_prob.transpose(0, 1)  # [B, K]
            elif action.ndim == 4:
                action = action.permute(1, 2, 0, 3)  # [B, S, K, dim_out]
                log_prob = log_prob.permute(1, 2, 0)  # [B, S, K]
            elif action.ndim == 5:
                action = action.permute(1, 2, 0, 3, 4)  # [B, S, K, L, dim_out]
                log_prob = log_prob.permute(1, 2, 0, 3)  # [B, S, K, L]
            else:
                raise ValueError(f"Unsupported action shape: {action.shape}")

        # action = dist.rsample()  # reparameterization trick

        return self.max_action * action, log_prob

    def log_prob(self, obs, actions):
        dist, _, _ = self.policy(obs)
        return dist.log_prob(actions / self.max_action).sum(-1)

    def entropy(self, obs):
        dist, _, _ = self.policy(obs)
        return dist.entropy().sum(-1)

    def covariance(self, obs):
        if self.contextual_cov:
            if self.std_only:
                std = self.log_std_multiplier * torch.exp(self.variance_net(obs)) + self.log_std_offset
            else:
                cov_vector = self.variance_net(obs)
                params_L = self._vector_to_cholesky(cov_vector)
                cov_matrix = params_L @ params_L.transpose(-1, -2)
                std = torch.sqrt(torch.diagonal(cov_matrix, dim1=-2, dim2=-1))
        else:
            std = self.log_std_multiplier * torch.exp(self.variance_net.variable) + self.log_std_offset
        return torch.diag_embed(std ** 2)

    def log_determinant(self, obs):
        cov = self.covariance(obs)
        return torch.logdet(cov)

    def precision(self, obs):
        cov = self.covariance(obs)
        return torch.inverse(cov)

    def maha(self, obs, actions):
        mean = self.mean_net(obs)
        diff = actions - mean
        precision = self.precision(obs)
        return torch.einsum("bi,bij,bj->b", diff, precision, diff)

    def expend_obs(self, obv, dim, repeat):
        return extend_and_repeat(obv, dim, repeat)
    @torch.no_grad()
    def act(self, obv, device):
        """
        Sample action with re-parametrization trick
        """
        obv = torch.tensor(obv.reshape(1, -1), device=self.device, dtype=self.dtype)
        action, _ = self.sample(obv)
        return action.cpu().data.numpy().flatten()

    def train(self):
        self.mean_net.train()
        if self.contextual_cov:
            self.variance_net.train()

    def eval(self):
        self.mean_net.eval()
        if self.contextual_cov:
            self.variance_net.eval()


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant

# CQL specific
def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)