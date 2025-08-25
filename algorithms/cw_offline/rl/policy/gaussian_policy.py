import torch
import torch.nn as nn
from torch.distributions import Normal

from algorithms.cw_offline.rl.policy import AbstractGaussianPolicy

class GaussianPolicy(AbstractGaussianPolicy):
    """
    Minimal unsquashed diagonal Gaussian policy for IQL/BC/AWAC.

    - Uses AbstractGaussianPolicy's mean_net (set its last act to tanh in config).
    - Uses a state-independent diagonal std (non-contextual variance).
    - policy(obs) -> (Normal(mean, std), mean, std)   # in model space
    - act(state) returns env-scale actions (multiplied by max_action).
    - log_prob(obs, actions) expects env-scale actions and divides by max_action.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        mean_net_args: dict,
        variance_net_args: dict,    # set contextual=False, std_only=True in the caller
        init_method: str,
        out_layer_gain: float,
        act_func_hidden: str,
        act_func_last: str,         # typically "tanh" to bound mean in (-1,1)
        max_action: float,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        dtype: str = "torch.float32",
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(
            dim_in, dim_out,
            mean_net_args, variance_net_args,
            init_method, out_layer_gain,
            act_func_hidden, act_func_last,
            dtype, device, **kwargs
        )
        self.max_action = float(max_action)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        # Ensure we are in the state-independent, diagonal-std regime
        assert not self.contextual_cov, "IQLGaussianPolicy expects contextual=False"
        assert self.std_only, "IQLGaussianPolicy expects std_only=True (diagonal std)"

        # Initialize log-std to zeros (like the simple GaussianPolicy)
        # variance_net is a TrainableVariable when contextual=False
        with torch.no_grad():
            self.variance_net.variable.data.zero_()

    def policy(self, obs: torch.Tensor):
        """
        Returns a Normal(mean, std) in model space (no tanh transform on the distribution).
        Mean is whatever mean_net outputs (often tanh-bounded if configured that way).
        """
        mean = self.mean_net(obs)
        log_std = self.variance_net.variable
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist, mean, std

    # -------- sampling / log-probs --------
    def sample(self, obs: torch.Tensor, num_samples: int = 1):
        """
        Sample reparameterized actions from the unsquashed Normal and return
        ENV-scale actions (multiplied by max_action) and their log_prob (in model space).
        """
        dist, mean, std = self.policy(obs)

        if num_samples == 1:
            if self.mean_net.training:
                action = dist.rsample()                      # [B, D]
                log_prob = dist.log_prob(action).sum(-1)     # [B]
            else:
                action = dist.mean
                log_prob = dist.log_prob(action).sum(-1)
            return self.max_action * action, log_prob

        # Optional: support K samples if requested
        action = dist.rsample((num_samples,))             # [K, B, D]
        log_prob = dist.log_prob(action).sum(-1)          # [K, B]
        action = action.transpose(0, 1)                   # [B, K, D]
        log_prob = log_prob.transpose(0, 1)               # [B, K]
        return self.max_action * action, log_prob

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        actions are ENV-scale; map back to model space by dividing by max_action.
        (Constant -d*log(max_action) is omitted; it doesn't affect gradients.)
        """
        dist, _, _ = self.policy(obs)
        a_model = actions / self.max_action
        return dist.log_prob(a_model).sum(-1)

    def entropy(self, obs: torch.Tensor):
        dist, _, _ = self.policy(obs)
        return dist.entropy().sum(-1)

    def covariance(self, obs: torch.Tensor):
        _dist, _mean, std = self.policy(obs)
        return torch.diag_embed(std ** 2)

    def log_determinant(self, obs: torch.Tensor):
        _dist, _mean, std = self.policy(obs)
        return (2 * torch.log(std + 1e-12)).sum(-1)

    def precision(self, obs: torch.Tensor):
        _dist, _mean, std = self.policy(obs)
        return torch.diag_embed(1.0 / (std ** 2 + 1e-12))

    def maha(self, obs: torch.Tensor, actions: torch.Tensor):
        dist, mean, std = self.policy(obs)
        diff = (actions / self.max_action) - mean
        return ((diff / (std + 1e-12)) ** 2).sum(-1)

    @torch.no_grad()
    def act(self, obv, device):
        obv = torch.tensor(obv.reshape(1, -1), device=self.device, dtype=self.dtype)
        action, _ = self.sample(obv)
        return action.cpu().numpy().flatten()

    def train(self):
        self.mean_net.train()
        # variance is a TrainableVariable; nothing to toggle

    def eval(self):
        self.mean_net.eval()