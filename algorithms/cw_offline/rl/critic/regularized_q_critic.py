import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.cw_offline import util
from algorithms.cw_offline.rl.critic import AbstractCritic
from algorithms.cw_offline.util import FullyConnectedQFunction


class RegularizedQCritic(AbstractCritic):
    def __init__(self, **config):
        self.single_q = config.get("single_q", False)
        self.config = config
        self.dtype, self.device = util.parse_dtype_device(config["dtype"],
                                                          config["device"])
        self.net1 = None
        self.net2 = None
        self.target_net1 = None
        self.target_net2 = None
        self.eta = config.get("update_rate", 0.005)
        self._create_network()

    def _create_network(self):
        """
        Create critic net with given configuration

        @return:
            None
        """
        config1 = copy.deepcopy(self.config)
        config1["name"] = self._critic_net_type + "_1"
        config2 = copy.deepcopy(self.config)
        config2["name"] = self._critic_net_type + "_2"

        self.net1 = FullyConnectedQFunction(**config1).to(self.device)
        self.net1.train()
        self.target_net1 = copy.deepcopy(self.net1).eval().requires_grad_(False)
        self.target_net1.eval()

        if not self.single_q:
            self.net2 = FullyConnectedQFunction(**config2).to(self.device)
            self.net2.train()
            self.target_net2 = copy.deepcopy(self.net2).eval().requires_grad_(False)
            self.target_net2.eval()
        else:
            self.net2 = self.net1
            self.target_net2 = self.target_net1

    def configure_optimizer(self, weight_decay, learning_rate, betas):
        """
        The optimizer is chosen to be AdamW
        @return: constructed optimizer
        """
        opt1 = self.net1.configure_optimizer(weight_decay=weight_decay,
                                             learning_rate=learning_rate,
                                             betas=betas,
                                             device_type=self.config["device"])

        if not self.single_q:
            opt2 = self.net2.configure_optimizer(weight_decay=weight_decay,
                                             learning_rate=learning_rate,
                                             betas=betas,
                                             device_type=self.config["device"])
        else:
            opt2 = opt1

        return opt1, opt2

    def critic(self, net, d_state, c_state, actions, idx_d, idx_c, idx_a):
        return net(d_state, c_state, actions, idx_d, idx_c, idx_a)

    def update_target_net(self, net, target_net):
        if self.single_q:
            assert self.net1 == self.net2
            assert self.target_net1 == self.target_net2

        for target_param, source_param in zip(target_net.parameters(),
                                              net.parameters()):
            target_param.data.copy_(self.eta * source_param.data
                                    + (1 - self.eta) * target_param.data)

    def parameters(self):
        """
        Get network parameters
        Returns:
            parameters
        """
        return list(self.net1.parameters()) + ([] if self.single_q else list(self.net2.parameters()))

    def train(self):
        self.net1.train()
        if not self.single_q:
            self.net2.train()

    def eval(self):
        self.net1.eval()
        if not self.single_q:
            self.net2.eval()

    def save_weights(self, log_dir: str, epoch: int):
        """
        Save NN weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """
        self.net1.save(log_dir, epoch)
        if not self.single_q:
            self.net2.save(log_dir, epoch)

    def load_weights(self, log_dir: str, epoch: int):
        """
        Load NN weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        self.net1.load(log_dir, epoch)
        self.net1.train()
        self.target_net1 = copy.deepcopy(self.net1)
        self.target_net1.requires_grad_(False)
        self.target_net1.eval()

        if not self.single_q:
            self.net2.load(log_dir, epoch)
            self.net2.train()
            self.target_net2 = copy.deepcopy(self.net2)
            self.target_net2.requires_grad_(False)
            self.target_net2.eval()
        else:
            self.net2 = self.net1
            self.target_net2 = self.target_net1

    def td_loss(self, observations, actions, next_observations, rewards, dones, alpha, actor):
        """
            Compute the TD loss (optionally supporting single or double Q networks).
        """
        with torch.no_grad():
            next_actions, next_log_pi = actor(next_observations)

            target_q1 = self.target_net1(next_observations, next_actions)
            if not self.single_q:
                target_q2 = self.target_net2(next_observations, next_actions)
                target_q = torch.min(target_q1, target_q2)
            else:
                target_q = target_q1

            if self.config.get("backup_entropy", True):
                target_q -= alpha * next_log_pi

            td_target = rewards + (1.0 - dones) * self.config["discount"] * target_q

        current_q1 = self.net1(observations, actions)
        current_q2 = self.net2(observations, actions) if not self.single_q else None

        qf1_loss = F.mse_loss(current_q1, td_target)
        qf2_loss = F.mse_loss(current_q2, td_target) if not self.single_q else None

        return qf1_loss, qf2_loss, current_q1, current_q2, td_target

    def regularization_loss(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement regularization_loss")


class CQLCritic(RegularizedQCritic):
    def __init__(self, **config):
        super().__init__(**config)
        self.temp = config.get("cql_temp", 1.0)
        self.min_q_weight = config.get("cql_alpha", 5.0)
        self.n_action_samples = config.get("cql_n_actions", 10)

    def regularization_loss(self, observations, critic_net, actor, q_values_data):
        # Sample actions uniformly
        batch_size = observations.size(0)
        action_dim = actor.action_dim
        random_actions = torch.empty(batch_size, self.n_action_samples, action_dim, device=self.device).uniform_(-1, 1)

        # Get actions from current policy
        repeated_obs = observations.unsqueeze(1).repeat(1, self.n_action_samples, 1).view(-1, observations.shape[-1])
        sampled_actions, log_probs = actor(repeated_obs)
        sampled_actions = sampled_actions.view(batch_size, self.n_action_samples, -1)
        log_probs = log_probs.view(batch_size, self.n_action_samples)

        q_rand = critic_net(observations, random_actions)
        q_pi = critic_net(observations, sampled_actions)

        cat_q = torch.cat([q_rand, q_pi], dim=1)  # [B, 2*N]
        logsumexp = torch.logsumexp(cat_q / self.temp, dim=1) * self.temp
        conservative_loss = (logsumexp - q_values_data).mean() * self.min_q_weight
        return conservative_loss


class IQLCritic(RegularizedQCritic):
    def __init__(self, **config):
        super().__init__(**config)
        self.expectile = config.get("expectile", 0.7)

        # Create value network V(s)
        self.value_net = FullyConnectedQFunction(
            observation_dim=config["observation_dim"],
            action_dim=0,  # no action input
            orthogonal_init=config.get("orthogonal_init", False),
            n_hidden_layers=config.get("q_n_hidden_layers", 3),
        ).to(self.device)

    def regularization_loss(self, observations, q_values_data, **kwargs):
        """
        Expectile regression loss between Q(s, a) and V(s)
        """
        v_values = self.value_net(observations, actions=None)
        diff = q_values_data - v_values
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        return (weight * diff ** 2).mean()