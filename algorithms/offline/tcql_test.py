# source: https://github.com/young-geng/CQL/tree/934b0e8354ca431d6c083c4e3a29df88d4b0a24d
# https://arxiv.org/pdf/2006.04779.pdf
import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal, TanhTransform, TransformedDistribution

import algorithms.cw_offline.rl.policy as pl  # import TanhGaussianPolicy
import algorithms.cw_offline.rl.replay_buffer as rb
import algorithms.cw_offline.rl.critic as crit
import algorithms.cw_offline.util as util

import algorithms.cw_offline.rl.critic.seq_critic as seq_critic

from torch.cuda.amp import GradScaler

from tqdm import tqdm

import os, torch
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.autograd.set_detect_anomaly(True)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False


TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    device: str = "cuda:1"
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e2)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 64  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    alpha_multiplier: float = 1.0  # Multiplier for alpha in loss
    use_automatic_entropy_tuning: bool = True  # Tune entropy
    backup_entropy: bool = False  # Use backup entropy
    policy_lr: float = 3e-5  # Policy learning rate
    qf_lr: float = 3e-4  # Critics learning rate
    soft_target_update_rate: float = 5e-3  # Target network update rate
    target_update_period: int = 1  # Frequency of target nets updates
    cql_n_actions: int = 10  # Number of sampled actions
    cql_importance_sample: bool = True  # Use importance sampling
    cql_lagrange: bool = False  # Use Lagrange version of CQL
    cql_target_action_gap: float = -1.0  # Action gap
    cql_temp: float = 1.0  # CQL temperature
    cql_alpha: float = 10.0  # Minimal Q weight
    cql_max_target_backup: bool = False  # Use max target backup
    cql_clip_diff_min: float = -np.inf  # Q-function lower loss clipping
    cql_clip_diff_max: float = np.inf  # Q-function upper loss clipping
    orthogonal_init: bool = True  # Orthogonal initialization
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    q_n_hidden_layers: int = 3  # Number of hidden layers in Q networks
    bc_steps: int = int(0)  # Number of BC steps at start
    reward_scale: float = 5.0  # Reward scale for normalization
    reward_bias: float = -1.0  # Reward bias for normalization
    policy_log_std_multiplier: float = 1.0  # Stochastic policy std multiplier
    project: str = "TORL"  # wandb project name
    group: str = "CQL-D4RL"  # wandb group name
    name: str = "CQL_ac"  # wandb run name

    # New parameters for segment-based critic update
    use_segment_critic_update: bool = False  # Use segment-based critic update
    epochs_critic: int = 3  # Number of critic update epochs
    segment_length: int = 8  # Length of segments for critic update (legacy, not used)
    num_segments: Union[int, str] = 5  # Number of segments (from segments_config) or "random"
    n_step_return: int = 4  # N-step return for critic update
    clip_grad_norm: float = 1.0  # Gradient clipping norm
    use_mix_precision: bool = False  # Use mixed precision training
    
    # New parameters for segment-based n-step return Q-learning
    use_segment_n_step_return_qf: bool = False  # Use segment n-step return Q-learning
    return_type: str = "segment_n_step_return_qf"  # Type of return computation

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
        env: gym.Env,
        state_mean: Union[np.ndarray, float] = 0.0,
        state_std: Union[np.ndarray, float] = 1.0,
        reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
                state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            buffer_size: int,
            device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
        seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


# @torch.no_grad()
# def eval_actor(
#         env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
# ) -> np.ndarray:
#     env.reset(seed=seed)
#     # actor.eval()
#     episode_rewards = []
#     for _ in range(n_episodes):
#         state, done = env.reset(), False
#         episode_reward = 0.0
#         while not done:
#             action = actor.act(state, device)
#             # action = actor.sample(state)
#             # print("action", action)
#             state, reward, done, _ = env.step(action)
#             episode_reward += reward
#             # print("reward", reward)
#         episode_rewards.append(episode_reward)
#
#     # actor.train()
#     return np.asarray(episode_rewards)

@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.reset(seed=seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(
        dataset: Dict,
        env_name: str,
        max_episode_steps: int = 1000,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def init_module_weights(module: torch.nn.Sequential, orthogonal_init: bool = False):
    # Specific orthgonal initialization for inner layers
    # If orthogonal init is off, we do not change default initialization
    if orthogonal_init:
        for submodule in module[:-1]:
            if isinstance(submodule, nn.Linear):
                nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
                nn.init.constant_(submodule.bias, 0.0)

    # Lasy layers should be initialzied differently as well
    if orthogonal_init:
        nn.init.orthogonal_(module[-1].weight, gain=1e-2)
    else:
        nn.init.xavier_uniform_(module[-1].weight, gain=1e-2)

    nn.init.constant_(module[-1].bias, 0.0)


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(
            self, log_std_min: float = -20.0, log_std_max: float = 2.0, no_tanh: bool = False
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(
            self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(
            self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob


class TanhGaussianPolicy(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            max_action: float,
            log_std_multiplier: float = 1.0,
            log_std_offset: float = -1.0,
            orthogonal_init: bool = False,
            no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )

        init_module_weights(self.base_network)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(
            self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        _, log_probs = self.tanh_gaussian(mean, log_std, False)
        return log_probs

    def forward(
            self,
            observations: torch.Tensor,
            deterministic: bool = False,
            repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)
        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        with torch.no_grad():
            actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class FullyConnectedQFunction(nn.Module):
    def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            orthogonal_init: bool = False,
            n_hidden_layers: int = 3,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init

        layers = [
            nn.Linear(observation_dim + action_dim, 256),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))

        self.network = nn.Sequential(*layers)

        init_module_weights(self.network, orthogonal_init)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        input_tensor = torch.cat([observations, actions], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class ContinuousCQL:
    def __init__(
            self,
            critic_1,
            # target_critic_1,
            critic_1_optimizer,
            critic_2,
            # target_critic_2,
            critic_2_optimizer,
            # critic, # modified for cql test
            actor,
            actor_optimizer,
            target_entropy: float,
            discount: float = 0.99,
            alpha_multiplier: float = 1.0,
            use_automatic_entropy_tuning: bool = True,
            backup_entropy: bool = False,
            policy_lr: bool = 3e-4,
            qf_lr: bool = 3e-4,
            soft_target_update_rate: float = 5e-3,
            bc_steps=100000,
            target_update_period: int = 1,
            cql_n_actions: int = 10,
            cql_importance_sample: bool = True,
            cql_lagrange: bool = False,
            cql_target_action_gap: float = -1.0,
            cql_temp: float = 1.0,
            cql_alpha: float = 5.0,
            cql_max_target_backup: bool = False,
            cql_clip_diff_min: float = -np.inf,
            cql_clip_diff_max: float = np.inf,
            device: str = "cpu",
    ):
        super().__init__()

        self.discount = discount
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.backup_entropy = backup_entropy
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.soft_target_update_rate = soft_target_update_rate
        self.bc_steps = bc_steps
        self.target_update_period = target_update_period
        self.cql_n_actions = cql_n_actions
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_alpha = cql_alpha
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self._device = device

        # New parameters for segment-based critic update
        self.use_segment_critic_update = False  # Use segment-based critic update
        self.epochs_critic = 3  # Number of critic update epochs
        self.num_segments = 5  # Number of segments (from segments_config) or "random"
        self.n_step_return = 4  # N-step return for critic update
        self.clip_grad_norm = 1.0  # Gradient clipping norm
        self.use_mix_precision = False  # Use mixed precision training
        
        # New parameters for segment-based n-step return Q-learning
        self.use_segment_n_step_return_qf = True  # Use segment n-step return Q-learning
        self.return_type = "segment_n_step_return_implicit_vf"  # Type of return computation
        # segment_n_step_return_qf
        # segment_n_step_return_implicit_vf
        self.num_samples_in_targets = 10  # Number of samples in targets for segment n-step return
        self.num_samples_in_policy = 1
        self.num_samples_in_cql_loss = 10  # Number of samples in CQL loss
        self.dtype, self.device = util.parse_dtype_device("torch.float32", "cuda:1")

        self.random_target = True  # Use random target for segment n-step return
        self.discount_factor = torch.tensor(float(1),
                                            dtype=self.dtype,
                                            device=self.device)  # Discount factor for segment n-step return
        self.targets_overestimate = False
        
        # Initialize traj_length (will be set during training)
        self.traj_length = None

        self.total_it = 0

        self.norm_data = False  # Normalize states

        self.critic_1 = critic_1
        self.critic_2 = critic_2

        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)
        # self.target_critic_1 = target_critic_1
        # self.target_critic_2 = target_critic_2

        self.actor = actor

        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer
        self.critic = None

        # self.critic = critic
        # # self.critic_1_optimizer, self.critic_2_optimizer = self.critic.configure_optimizer(
        # #     weight_decay=0.0,
        # #     learning_rate=self.qf_lr,
        # #     betas=(0.9, 0.999),
        # # )

        # self.wd_critic = 1e-5
        # self.lr_critic = 5e-5
        # self.betas = (0.9, 0.999)
        # self.critic_1_optimizer, self.critic_2_optimizer = self.critic.configure_optimizer(
        #     weight_decay=self.wd_critic, learning_rate=self.lr_critic,
        #     betas=self.betas)
        # self.critic_optimizer = (self.critic_1_optimizer, self.critic_2_optimizer)
        #
        # if not self.critic.single_q:
        #     self.critic_grad_scaler = [GradScaler(), GradScaler()]
        # else:
        #     self.critic_grad_scaler = [GradScaler()] * 2

        # critic_1, critic_2 = critic.net1, critic.net2
        # target_critic_1, target_critic_2 = critic.target_net1, critic.target_net2
        # critic_1_optimizer, critic_2_optimizer = critic.configure_optimizer(
        #     weight_decay=0.0,
        #     learning_rate=config.qf_lr,
        #     betas=(0.9, 0.999),
        # )

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.policy_lr,
            )
        else:
            self.log_alpha = None

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.qf_lr,
        )

        self.total_it = 0

        self.progress_bar = tqdm(total=1e6)

        self.iql_tau = 0.7


    def get_segments(self, pad_additional=False):
        """Generate segment indices for critic update.
        
        Args:
            pad_additional (bool): Whether to add an additional segment if needed
            
        Returns:
            torch.Tensor: Segment indices with shape [num_segments, segment_length + 1]
        """
        # Get num_segments from config
        num_seg = self.num_segments
        if isinstance(num_seg, int):
            pass
        elif isinstance(num_seg, str):
            if num_seg == "random":
                possible_num_segments = torch.arange(1, 26, device=self._device)
                segment_lengths = self.traj_length // possible_num_segments
                segment_lengths_unique = segment_lengths.unique()
                possible_num_segments_after_unique = self.traj_length // segment_lengths_unique
                # Random choose the number of segments
                num_seg = possible_num_segments_after_unique[torch.randint(
                    0, len(possible_num_segments_after_unique), [], 
                    device=self._device)]
            else:
                raise ValueError("Invalid num_seg")
        
        seg_length = self.traj_length // num_seg
        if num_seg == 1:
            start_idx = 0
        else:
            start_idx = 0

            # start_idx = torch.randint(0, seg_length, [],
            #                           dtype=torch.long, device=self._device)
        
        num_seg_actual = (self.traj_length - start_idx) // seg_length
        if pad_additional:
            num_seg_actual += 1
        
        idx_in_segments = torch.arange(0, num_seg_actual * seg_length,
                                       device=self._device) + start_idx
        idx_in_segments = idx_in_segments.view(-1, seg_length)
        idx_in_segments = torch.cat([idx_in_segments,
                                     idx_in_segments[:, -1:] + 1], dim=-1)
        
        if pad_additional and idx_in_segments[-1][0] == self.traj_length:
            return idx_in_segments[:-1]
        else:
            return idx_in_segments

    # def update_target_network(self, soft_target_update_rate: float):
    #     soft_update(self.critic.target_net1, self.critic.net1, soft_target_update_rate)
    #     soft_update(self.critic.target_net2, self.critic.net2, soft_target_update_rate)

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic_1, self.critic_1, soft_target_update_rate)
        soft_update(self.target_critic_2, self.critic_2, soft_target_update_rate)

    def _alpha_and_alpha_loss(self, observations: torch.Tensor, log_pi: torch.Tensor):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                    self.log_alpha() * (log_pi + self.target_entropy).detach()
            ).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    # def _policy_loss(
    #         self,
    #         observations: torch.Tensor,
    #         actions: torch.Tensor,
    #         new_actions: torch.Tensor,
    #         alpha: torch.Tensor,
    #         log_pi: torch.Tensor,
    # ) -> torch.Tensor:
    #     self.critic.eval()  # disable dropout
    #     self.critic.requires_grad(False)
    #
    #     # reshape observations and actions, generate idx for transformer critic
    #     observations = observations.unsqueeze(1)
    #     idx_c = torch.ones((observations.shape[0], observations.shape[1],), dtype=torch.int64, device=self._device)
    #     if new_actions.ndim == 2:
    #         new_actions = new_actions.unsqueeze(1).unsqueeze(1)
    #         idx_a = torch.ones((observations.shape[0], observations.shape[1], new_actions.shape[1]), dtype=torch.int64, device=self._device)
    #     else:
    #         # for action_sequence, not checked yet
    #         idx_in_segments = self.get_segments()
    #         num_segments = idx_in_segments.shape[0]
    #         idx_a = idx_in_segments[..., :-1]
    #
    #
    #     if self.total_it <= self.bc_steps:
    #         log_probs = self.actor.log_prob(observations, actions)
    #         policy_loss = (alpha * log_pi - log_probs).mean()
    #     else:
    #         # Use new actions for policy loss
    #         # with torch.no_grad():
    #         q1 = self.critic.critic(
    #                 net=self.critic.net1, d_state=None, c_state=observations,
    #                 actions=new_actions, idx_d=None, idx_c=idx_c,
    #                 idx_a=idx_a)[..., 1:]
    #         if not self.critic.single_q:
    #             q2 =self.critic.critic(
    #                     net=self.critic.net2, d_state=None, c_state=observations,
    #                     actions=new_actions, idx_d=None, idx_c=idx_c,
    #                     idx_a=idx_a)[..., 1:]
    #         else:
    #             q2 = q1
    #         q_new_actions = torch.minimum(q1, q2)
    #
    #         q_new_actions = q_new_actions.squeeze()
    #
    #         policy_loss = (alpha * log_pi - q_new_actions).mean()
    #         # TODO: add entropy loss and trust region loss to policy loss
    #     return policy_loss

    def _policy_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        new_actions: torch.Tensor,
        alpha: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        if self.total_it <= self.bc_steps:
            log_probs = self.actor.log_prob(observations, actions)
            policy_loss = (alpha * log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(
                self.critic_1(observations, new_actions),
                self.critic_2(observations, new_actions),
            )
            policy_loss = (alpha * log_pi - q_new_actions).mean()
        return policy_loss

    # def _q_loss(
    #         self,
    #         observations: torch.Tensor,
    #         actions: torch.Tensor,
    #         next_observations: torch.Tensor,
    #         rewards: torch.Tensor,
    #         dones: torch.Tensor,
    #         alpha: torch.Tensor,
    #         log_dict: Dict,
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     # 1. TD loss
    #     qf1_loss, qf2_loss, q1_predicted, q2_predicted, td_target, target_q_values = self.critic.td_loss(
    #         observations, actions, next_observations, rewards, dones, alpha, self.actor
    #     )
    #
    #     # 2. CQL loss
    #     conservative_loss_1, std_q1, q1_rand, q1_curr_actions, q1_next_actions = (
    #         self.critic.regularization_loss(
    #             observations, next_observations, self.critic.net1, self.actor, q1_predicted
    #         ))
    #     conservative_loss_2, std_q2, q2_rand, q2_curr_actions, q2_next_actions = (
    #         self.critic.regularization_loss(
    #             observations, next_observations, self.critic.net2, self.actor, q2_predicted
    #         ))
    #
    #     # """Subtract the log likelihood of data"""
    #     # cql_qf1_diff = torch.clamp(
    #     #     cql_qf1_ood - q1_predicted,
    #     #     self.cql_clip_diff_min,
    #     #     self.cql_clip_diff_max,
    #     # ).mean()
    #     # cql_qf2_diff = torch.clamp(
    #     #     cql_qf2_ood - q2_predicted,
    #     #     self.cql_clip_diff_min,
    #     #     self.cql_clip_diff_max,
    #     # ).mean()
    #     # TODO: check if clamp necessary
    #
    #     # 3. Optional Lagrange version of CQL
    #     if self.cql_lagrange:
    #         alpha_prime = torch.clamp(
    #             torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
    #         )
    #         cql_min_qf1_loss = (
    #                 alpha_prime
    #                 * self.cql_alpha
    #                 * (conservative_loss_1 - self.cql_target_action_gap)
    #         )
    #         cql_min_qf2_loss = (
    #                 alpha_prime
    #                 * self.cql_alpha
    #                 * (conservative_loss_2 - self.cql_target_action_gap)
    #         )
    #
    #         self.alpha_prime_optimizer.zero_grad()
    #         alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
    #         alpha_prime_loss.backward(retain_graph=True)
    #         self.alpha_prime_optimizer.step()
    #     else:
    #         cql_min_qf1_loss = conservative_loss_1 * self.cql_alpha
    #         cql_min_qf2_loss = conservative_loss_2 * self.cql_alpha
    #         alpha_prime_loss = observations.new_tensor(0.0)
    #         alpha_prime = observations.new_tensor(0.0)
    #
    #     # 4. Final total Q-function loss
    #     qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss
    #
    #     # 5. Log values
    #     log_dict.update(
    #         dict(
    #             qf1_loss=qf1_loss.item(),
    #             qf2_loss=qf2_loss.item(),
    #             alpha=alpha.item(),
    #             average_qf1=q1_predicted.mean().item(),
    #             average_qf2=q2_predicted.mean().item(),
    #             average_target_q=target_q_values.mean().item(),
    #         )
    #     )
    #
    #     log_dict.update(
    #         dict(
    #             cql_std_q1=std_q1.mean().item(),
    #             cql_std_q2=std_q2.mean().item(),
    #             cql_q1_rand=q1_rand.mean().item(),
    #             cql_q2_rand=q2_rand.mean().item(),
    #             cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
    #             cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
    #             cql_qf1_diff=conservative_loss_1.mean().item(),
    #             cql_qf2_diff=conservative_loss_2.mean().item(),
    #             cql_q1_current_actions=q1_curr_actions.mean().item(),
    #             cql_q2_current_actions=q2_curr_actions.mean().item(),
    #             cql_q1_next_actions=q1_next_actions.mean().item(),
    #             cql_q2_next_actions=q2_next_actions.mean().item(),
    #             alpha_prime_loss=alpha_prime_loss.item(),
    #             alpha_prime=alpha_prime.item(),
    #         )
    #     )
    #
    #     return qf_loss, alpha_prime, alpha_prime_loss

    def _q_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        alpha: torch.Tensor,
        log_dict: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q1_predicted = self.critic_1(observations, actions)
        q2_predicted = self.critic_2(observations, actions)

        if self.cql_max_target_backup:
            new_next_actions, next_log_pi = self.actor(
                next_observations, repeat=self.cql_n_actions
            )
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_critic_1(next_observations, new_next_actions),
                    self.target_critic_2(next_observations, new_next_actions),
                ),
                dim=-1,
            )
            next_log_pi = torch.gather(
                next_log_pi, -1, max_target_indices.unsqueeze(-1)
            ).squeeze(-1)
        else:
            new_next_actions, next_log_pi = self.actor(next_observations)
            target_q_values = torch.min(
                self.target_critic_1(next_observations, new_next_actions),
                self.target_critic_2(next_observations, new_next_actions),
            )

        if self.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        target_q_values = target_q_values.unsqueeze(-1)
        td_target = rewards + (1.0 - dones) * self.discount * target_q_values.detach()
        td_target = td_target.squeeze(-1)
        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())

        # CQL
        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]
        cql_random_actions = actions.new_empty(
            (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis = self.actor(
            observations, repeat=self.cql_n_actions
        )
        cql_next_actions, cql_next_log_pis = self.actor(
            next_observations, repeat=self.cql_n_actions
        )
        cql_current_actions, cql_current_log_pis = (
            cql_current_actions.detach(),
            cql_current_log_pis.detach(),
        )
        cql_next_actions, cql_next_log_pis = (
            cql_next_actions.detach(),
            cql_next_log_pis.detach(),
        )

        cql_q1_rand = self.critic_1(observations, cql_random_actions)
        cql_q2_rand = self.critic_2(observations, cql_random_actions)
        cql_q1_current_actions = self.critic_1(observations, cql_current_actions)
        cql_q2_current_actions = self.critic_2(observations, cql_current_actions)
        cql_q1_next_actions = self.critic_1(observations, cql_next_actions)
        cql_q2_next_actions = self.critic_2(observations, cql_next_actions)

        cql_cat_q1 = torch.cat(
            [
                cql_q1_rand,
                torch.unsqueeze(q1_predicted, 1),
                cql_q1_next_actions,
                cql_q1_current_actions,
            ],
            dim=1,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_q2_rand,
                torch.unsqueeze(q2_predicted, 1),
                cql_q2_next_actions,
                cql_q2_current_actions,
            ],
            dim=1,
        )
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5**action_dim)
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_actions - cql_next_log_pis.detach(),
                    cql_q1_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q2_next_actions - cql_next_log_pis.detach(),
                    cql_q2_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q2_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
            )
            cql_min_qf1_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf1_diff - self.cql_target_action_gap)
            )
            cql_min_qf2_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_qf2_diff - self.cql_target_action_gap)
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
            cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha
            alpha_prime_loss = observations.new_tensor(0.0)
            alpha_prime = observations.new_tensor(0.0)

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss
        # print("QF Loss:", qf1_loss.item(), qf2_loss.item(), cql_min_qf1_loss.item(),
        #       cql_min_qf2_loss.item())

        log_dict.update(
            dict(
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha=alpha.item(),
                average_qf1=q1_predicted.mean().item(),
                average_qf2=q2_predicted.mean().item(),
                average_target_q=target_q_values.mean().item(),
                average_reward=rewards.mean().item(),
            )
        )

        log_dict.update(
            dict(
                cql_std_q1=cql_std_q1.mean().item(),
                cql_std_q2=cql_std_q2.mean().item(),
                cql_q1_rand=cql_q1_rand.mean().item(),
                cql_q2_rand=cql_q2_rand.mean().item(),
                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                cql_qf1_diff=cql_qf1_diff.mean().item(),
                cql_qf2_diff=cql_qf2_diff.mean().item(),
                cql_q1_current_actions=cql_q1_current_actions.mean().item(),
                cql_q2_current_actions=cql_q2_current_actions.mean().item(),
                cql_q1_next_actions=cql_q1_next_actions.mean().item(),
                cql_q2_next_actions=cql_q2_next_actions.mean().item(),
                alpha_prime_loss=alpha_prime_loss.item(),
                alpha_prime=alpha_prime.item(),
            )
        )

        return qf_loss, alpha_prime, alpha_prime_loss

    def _transformer_q_loss(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            next_observations: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            alpha: torch.Tensor,
            log_dict: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. TD loss
        qf1_loss, qf2_loss, q1_predicted, q2_predicted, td_target, target_q_values = self.critic.td_loss(
            observations, actions, next_observations, rewards, dones, alpha, self.actor
        )

        # 2. CQL loss
        conservative_loss_1, std_q1, q1_rand, q1_curr_actions, q1_next_actions = (
            self.critic.regularization_loss(
                observations, next_observations, self.critic.net1, self.actor, q1_predicted
            ))
        conservative_loss_2, std_q2, q2_rand, q2_curr_actions, q2_next_actions = (
            self.critic.regularization_loss(
                observations, next_observations, self.critic.net2, self.actor, q2_predicted
            ))

        # """Subtract the log likelihood of data"""
        # cql_qf1_diff = torch.clamp(
        #     cql_qf1_ood - q1_predicted,
        #     self.cql_clip_diff_min,
        #     self.cql_clip_diff_max,
        # ).mean()
        # cql_qf2_diff = torch.clamp(
        #     cql_qf2_ood - q2_predicted,
        #     self.cql_clip_diff_min,
        #     self.cql_clip_diff_max,
        # ).mean()
        # TODO: check if clamp necessary

        # 3. Optional Lagrange version of CQL
        if self.cql_lagrange:
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
            )
            cql_min_qf1_loss = (
                    alpha_prime
                    * self.cql_alpha
                    * (conservative_loss_1 - self.cql_target_action_gap)
            )
            cql_min_qf2_loss = (
                    alpha_prime
                    * self.cql_alpha
                    * (conservative_loss_2 - self.cql_target_action_gap)
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = conservative_loss_1 * self.cql_alpha
            cql_min_qf2_loss = conservative_loss_2 * self.cql_alpha
            alpha_prime_loss = observations.new_tensor(0.0)
            alpha_prime = observations.new_tensor(0.0)

        # 4. Final total Q-function loss
        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        # 5. Log values
        log_dict.update(
            dict(
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha=alpha.item(),
                average_qf1=q1_predicted.mean().item(),
                average_qf2=q2_predicted.mean().item(),
                average_target_q=target_q_values.mean().item(),
            )
        )

        log_dict.update(
            dict(
                cql_std_q1=std_q1.mean().item(),
                cql_std_q2=std_q2.mean().item(),
                cql_q1_rand=q1_rand.mean().item(),
                cql_q2_rand=q2_rand.mean().item(),
                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                cql_qf1_diff=conservative_loss_1.mean().item(),
                cql_qf2_diff=conservative_loss_2.mean().item(),
                cql_q1_current_actions=q1_curr_actions.mean().item(),
                cql_q2_current_actions=q2_curr_actions.mean().item(),
                cql_q1_next_actions=q1_next_actions.mean().item(),
                cql_q2_next_actions=q2_next_actions.mean().item(),
                alpha_prime_loss=alpha_prime_loss.item(),
                alpha_prime=alpha_prime.item(),
            )
        )

        return qf_loss, alpha_prime, alpha_prime_loss

    def segments_n_step_return_vf(self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            idx_in_segments: torch.Tensor,):
        """
        Segment-wise n-step return using Value function
        Use Q-func as the target of the V-func prediction
        Use N-step return + V-func as the target of the Q-func prediction
        Args:
            dataset:
            idx_in_segments:

        Returns:
            n_step_returns: [num_traj, num_segments, 1 + num_seg_actions]

        """
        states = observations  # [num_traj, traj_length, dim_state]

        # rewards, dones, actions stay the same
        # rewards = dataset["step_rewards"]
        # dones = dataset["step_dones"]
        # decision_idx = dataset["decision_idx"]  # [num_traj]
        # # [num_traj, traj_length, dim_action]
        # traj_init_pos = dataset["step_desired_pos"]
        # traj_init_vel = dataset["step_desired_vel"]

        num_segments = idx_in_segments.shape[0]
        num_seg_actions = idx_in_segments.shape[-1] - 1
        seg_start_idx = idx_in_segments[..., 0]

        num_traj, traj_length = states.shape[0], states.shape[1]

        # NOTE: additional dimension is defined as [num_traj, num_segments]
        # TODO: MP actor related part removed, action sampling is also not necessary? use action in the dataset or directly V(s_t+n)?

        # # [num_traj, num_segments, num_seg_actions, dim_action] or
        # # [num_traj, num_segments, num_smps, num_seg_actions, dim_action]
        # actions = self.policy.sample(require_grad=False,
        #                              params_mean=params_mean_new,
        #                              params_L=params_L_new, times=action_times,
        #                              init_time=init_time,
        #                              init_pos=init_pos, init_vel=init_vel,
        #                              use_mean=False,
        #                              num_samples=self.num_samples_in_targets)
        #
        # # Normalize actions
        # if self.norm_data:
        #     actions = self.replay_buffer.normalize_data("step_actions", actions)

        ##################### Compute the Q in the future ######################
        # [num_traj, num_segments, 1 + num_seg_actions]
        future_returns = torch.zeros([num_traj, num_segments,
                                      1 + num_seg_actions], device=self.device)

        # [num_traj, dim_state] -> [num_traj, num_segments, dim_state]
        # d_state = states[
        #     torch.arange(num_traj, device=self.device), decision_idx]
        # d_state = util.add_expand_dim(d_state, [1], [num_segments])

        # [num_traj] -> [num_traj, num_segments]
        # d_idx = util.add_expand_dim(decision_idx, [1], [num_segments])

        # [num_traj, num_segments, dim_state]
        # c_state = states[:, seg_start_idx]
        #
        # if self.num_samples_in_targets == 1:
        #     # [num_traj, num_segments]
        #     c_idx = util.add_expand_dim(seg_start_idx, [0], [num_traj])
        #
        #     # [num_segments, num_seg_actions]
        #     # -> [num_traj, num_segments, num_seg_actions]
        #     a_idx = util.add_expand_dim(idx_in_segments[..., :-1],
        #                                 [0], [num_traj])
        # else:
        #     num_smp = self.num_samples_in_targets
        #
        #     # -> [num_traj, num_segments, num_smp, dim_state]
        #     c_state = util.add_expand_dim(c_state, [2], [num_smp])
        #
        #     # [num_traj, num_segments, num_smp]
        #     c_idx = util.add_expand_dim(seg_start_idx, [0, 2],
        #                                 [num_traj, num_smp])
        #     # [num_segments, num_seg_actions]
        #     # -> [num_traj, num_segments, num_smp, num_seg_actions]
        #     a_idx = util.add_expand_dim(idx_in_segments[..., :-1],
        #                                 [0, 2],
        #                                 [num_traj, num_smp])
        #
        # # Use mix precision for faster computation
        # with util.autocast_if(self.use_mix_precision):
        #     # [num_traj, num_segments, (num_smp,) 1 + num_seg_actions]
        #     future_q1 = self.critic.critic(self.critic.target_net1,
        #                                    d_state=None, c_state=c_state,
        #                                    actions=acitons, idx_d=None,
        #                                    idx_c=c_idx, idx_a=a_idx)
        #     if not self.critic.single_q:
        #         future_q2 = self.critic.critic(self.critic.target_net2,
        #                                        d_state=None, c_state=c_state,
        #                                        actions=acitons, idx_d=None,
        #                                        idx_c=c_idx, idx_a=a_idx)
        #     else:
        #         future_q2 = future_q1
        #
        # # if self.random_target and not self.targets_overestimate:
        # if self.random_target:
        #     # Randomly choose the Q value from Q1 or Q2
        #     mask = torch.randint(0, 2, future_q1.shape, device=self.device)
        #     future_q = future_q1 * mask + future_q2 * (1 - mask)
        # else:
        #     future_q = torch.minimum(future_q1, future_q2)
        #
        # if self.num_samples_in_targets > 1:
        #     future_q = future_q.mean(dim=-2)

        # # Use last q as the target of the V-func
        # # [num_traj, num_segments, 1 + num_seg_actions]
        # # -> [num_traj, num_segments]
        # # Tackle the Q-func beyond the length of the trajectory
        # # Option, Use the last q predictions
        # future_returns[:, :-1, 0] = future_q[:, :-1, -1]
        #
        # # Find the idx where the action is the last action in the valid trajectory
        # last_valid_q_idx = idx_in_segments[-1] == traj_length
        # future_returns[:, -1, 0] = (
        #     future_q[:, -1, last_valid_q_idx].squeeze(-1))
        #
        # # Option, Use the average q predictions, exclude the vf at 0th place
        # # Option does not work well
        # # future_returns[:, :-1, 0] = future_q[:, :-1, 1:].mean(dim=-1)
        #
        # # Find the idx where the action is the last action in the valid trajectory
        # # last_valid_q_idx = (idx_in_segments[-1] <= traj_length)[1:]
        # # future_returns[:, -1, 0] = future_q[:, -1, 1:][..., last_valid_q_idx].mean(dim=-1)

        # TODO: check how to get the target of the first column V(s0)


        ##################### Compute the V in the future ######################
        # state after executing the action
        # [num_traj, traj_length]
        c_idx = torch.arange(traj_length, device=self.device).long()
        c_idx = util.add_expand_dim(c_idx, [0], [num_traj])

        # [num_traj, traj_length, dim_state]
        c_state = states

        # Use mix precision for faster computation
        with util.autocast_if(self.use_mix_precision):
            # [num_traj, traj_length]
            future_v1 = self.critic.critic(self.critic.target_net1,
                                           d_state=None, c_state=c_state,
                                           actions=None, idx_d=None,
                                           idx_c=c_idx, idx_a=None).squeeze(-1)
            if not self.critic.single_q:
                future_v2 = self.critic.critic(self.critic.target_net2,
                                               d_state=None, c_state=c_state,
                                               actions=None, idx_d=None,
                                               idx_c=c_idx, idx_a=None).squeeze(-1)
            else:
                future_v2 = future_v1

        # if self.random_target and not self.targets_overestimate:
        if self.random_target:
            # Randomly choose the V value from V1 or V2
            mask = torch.randint(0, 2, future_v1.shape, device=self.device)
            future_v = future_v1 * mask + future_v2 * (1 - mask)
        else:
            future_v = torch.minimum(future_v1, future_v2)

        # Pad zeros to the states which go beyond the traj length
        future_v_pad_zero_end \
            = torch.nn.functional.pad(future_v, (0, num_seg_actions))

        # [num_segments, num_seg_actions]
        v_idx = idx_in_segments[:, 1:]
        # assert v_idx.max() <= traj_length

        # [num_traj, traj_length] -> [num_traj, num_segments, num_seg_actions]
        future_returns[..., 1:] = future_v_pad_zero_end[:, v_idx]


        ################ Compute the reward in the future ##################
        # [num_traj, traj_length] -> [num_traj, traj_length + padding]
        future_r_pad_zero_end \
            = torch.nn.functional.pad(rewards, (0, num_seg_actions))

        # discount_seq as: [1, gamma, gamma^2..., 0]
        discount_idx \
            = torch.arange(traj_length + num_seg_actions, device=self.device)
        discount_seq = self.discount_factor.pow(discount_idx)
        discount_seq[-num_seg_actions:] = 0

        # Apply discount to all rewards and returns w.r.t the traj start
        # [num_traj, num_segments, 1 + traj_length]
        discount_returns = future_returns * discount_seq[idx_in_segments]

        # [num_traj, traj_length + 1]
        discount_r = future_r_pad_zero_end * discount_seq

        # -> [num_traj, num_segments, 1 + traj_length]
        discount_r = util.add_expand_dim(discount_r, [1],
                                         [num_segments])

        # [num_traj, num_segments, 1 + num_seg_actions]
        seg_discount_q = discount_returns

        # -> [num_traj, num_segments, 1 + num_seg_actions]
        seg_reward_idx = util.add_expand_dim(idx_in_segments, [0],
                                             [num_traj])

        # torch.gather shapes
        # input: [num_traj, num_segments, traj_length + 1]
        # index: [num_traj, num_segments, 1 + num_seg_actions]
        # result: [num_traj, num_segments, 1 + num_seg_actions]
        seg_discount_r = torch.gather(input=discount_r, dim=-1,
                                      index=seg_reward_idx)

        # [num_traj, num_segments, 1 + num_seg_actions] ->
        # [num_traj, num_segments, 1 + num_seg_actions1, 1 + num_seg_actions]
        seg_discount_r = util.add_expand_dim(seg_discount_r, [-2],
                                             [1 + num_seg_actions])

        # Get a lower triangular mask with off-diagonal elements as 1
        # [num_seg_actions + 1, num_seg_actions + 1]
        reward_tril_mask = torch.tril(torch.ones(1 + num_seg_actions,
                                                 1 + num_seg_actions,
                                                 device=self.device),
                                      diagonal=-1)

        # [num_traj, num_segments, num_seg_actions + 1, num_seg_actions + 1]
        tril_discount_rewards = seg_discount_r * reward_tril_mask

        discount_start \
            = self.discount_factor.pow(seg_start_idx)[None, :, None]

        # N-step return as target
        # V(s0) -> R0
        # Q(s0, a0) -> r0 + \gam * R1
        # Q(s0, a0, a1) -> r0 + \gam * r1 + \gam^2 * R2
        # Q(s0, a0, a1, a2) -> r0 + \gam * r1 + \gam^2 * r2 + \gam^3 * R3

        # [num_traj, num_segments, 1 + num_seg_actions]
        n_step_returns = (tril_discount_rewards.sum(
            dim=-1) + seg_discount_q) / discount_start

        ####################################################################

        return n_step_returns

    def segments_n_step_return_qf(self,
                                  observations: torch.Tensor,
                                  actions: torch.Tensor,
                                  next_observations: torch.Tensor,
                                  rewards: torch.Tensor,
                                  dones: torch.Tensor,
                                  idx_in_segments: torch.Tensor, ):
        """
                Segment-wise n-step return using Q function
                Use N-step return + Expectation of Q-func as the target of the Q-func prediction
                Args:
                    dataset:
                    idx_in_segments:

                Returns:
                    n_step_returns: [num_traj, num_segments, 1 + num_seg_actions]

                """
        states = observations  # [num_traj, traj_length, dim_state]

        num_segments = idx_in_segments.shape[0]
        num_seg_actions = idx_in_segments.shape[-1] - 1
        seg_start_idx = idx_in_segments[..., 0]

        num_traj, traj_length = states.shape[0], states.shape[1]

        # TODO: check if the index is correct
        n_idx = torch.arange(traj_length, device=self.device).long()
        n_idx = util.add_expand_dim(n_idx, [0], [num_traj])

        a_idx = util.add_expand_dim(n_idx, [-1], [1])

        n_states = next_observations[:, seg_start_idx]  # [num_traj, num_segments, dim_state]
        n_states_seq = next_observations

        # [num_traj, num_segments, 1, dim_action] or
        # [num_traj, num_segments, num_smps, 1, dim_action]
        # For step-based actor, only sample action based on the initial state
        with torch.no_grad():
            n_actions, n_action_log_pi = self.actor.sample(n_states_seq, num_samples=self.num_samples_in_targets)
            # n_actions, n_action_log_pi = self.actor(n_states_seq)

        # Normalize actions
        if self.norm_data:
            n_actions = self.replay_buffer.normalize_data("step_actions", n_actions)

        n_actions = n_actions.unsqueeze(-2)  # [num_traj, num_segments, (num_smps), 1, dim_action]

        ##################### Compute the Q(s', a') based on the sampled actions ######################
        # [num_traj, num_segments, 1 + num_seg_actions]
        future_returns = torch.zeros([num_traj, num_segments,
                                      1 + num_seg_actions], device=self.device)

        # [num_traj, dim_state] -> [num_traj, num_segments, dim_state]
        # d_state = states[
        #     torch.arange(num_traj, device=self.device), decision_idx]
        # d_state = util.add_expand_dim(d_state, [1], [num_segments])

        # [num_traj] -> [num_traj, num_segments]
        # d_idx = util.add_expand_dim(decision_idx, [1], [num_segments])

        # [num_traj, num_segments, dim_state]
        # c_state = states[:, seg_start_idx]
        #
        # sample the first index
        # if self.num_samples_in_targets == 1:
        #     # [num_traj, num_segments]
        #     n_idx = util.add_expand_dim(seg_start_idx+1, [0], [num_traj])
        #
        #     # [num_segments, num_seg_actions]
        #     # -> [num_traj, num_segments, num_seg_actions]
        #     a_idx = util.add_expand_dim(seg_start_idx+1,
        #                                 [0, 2], [num_traj, 1])
        # else:
        #     num_smp = self.num_samples_in_targets
        #
        #     # -> [num_traj, num_segments, num_smp, dim_state]
        #     n_states = util.add_expand_dim(n_states, [2], [num_smp])
        #
        #     # [num_traj, num_segments, num_smp]
        #     n_idx = util.add_expand_dim(seg_start_idx, [0, 2],
        #                                 [num_traj, num_smp])
        #     # [num_segments, num_seg_actions]
        #     # -> [num_traj, num_segments, num_smp, num_seg_actions]
        #     a_idx = util.add_expand_dim(seg_start_idx+1,
        #                                 [0, 2, 3],
        #                                 [num_traj, num_smp, 1])

        # sample the entire trajectory
        # if self.num_samples_in_targets == 1:
        #     # [num_traj, num_segments]
        #     n_idx = util.add_expand_dim(seg_start_idx+1, [0], [num_traj])
        #
        #     # [num_segments, num_seg_actions]
        #     # -> [num_traj, num_segments, num_seg_actions]
        #     a_idx = util.add_expand_dim(seg_start_idx+1,
        #                                 [0, 2], [num_traj, 1])
        if self.num_samples_in_targets != 1:
            num_smp = self.num_samples_in_targets

            # -> [num_traj, traj_length, num_smp, dim_state]
            n_states_seq = util.add_expand_dim(n_states_seq, [2], [num_smp])

            # [num_traj, traj_length, num_smp]
            n_idx = util.add_expand_dim(n_idx, [-1],
                                        [num_smp])
            # [num_segments, num_seg_actions]
            # -> [num_traj, traj_length, num_smp, num_seg_actions]
            a_idx = util.add_expand_dim(a_idx,
                                        [2],
                                        [num_smp])

        # Use mix precision for faster computation
        with util.autocast_if(self.use_mix_precision):
            # [num_traj, num_segments, (num_smp,) 1 + num_seg_actions]
            future_q1_sampled = self.critic.critic(self.critic.target_net1,
                                           d_state=None, c_state=n_states_seq,
                                           actions=n_actions, idx_d=None,
                                           idx_c=n_idx, idx_a=a_idx)
            if not self.critic.single_q:
                future_q2_sampled = self.critic.critic(self.critic.target_net2,
                                               d_state=None, c_state=n_states_seq,
                                               actions=n_actions, idx_d=None,
                                               idx_c=n_idx, idx_a=a_idx)
            else:
                future_q2_sampled = future_q1_sampled

        # if self.random_target and not self.targets_overestimate:
        if self.random_target:
            # Randomly choose the Q value from Q1 or Q2
            mask = torch.randint(0, 2, future_q1_sampled.shape, device=self.device)
            min_q1_q2 = future_q1_sampled * mask + future_q2_sampled * (1 - mask)
        else:
            min_q1_q2 = torch.minimum(future_q1_sampled, future_q2_sampled)

        if self.num_samples_in_targets > 1:
            min_q1_q2 = min_q1_q2.mean(dim=-2)

        # min_q1_q2 = min_q1_q2[..., 1].squeeze()

        future_q_sampled = torch.zeros([num_traj, traj_length],
                               device=self.device)
        future_q_sampled = min_q1_q2[..., 1].squeeze()

        # Pad zeros to the states which go beyond the traj length
        future_q_pad_zero_end \
            = torch.nn.functional.pad(future_q_sampled, (0, num_seg_actions))

        # [num_segments, num_seg_actions]
        q_idx = idx_in_segments[:, :-1]  # since next_states used
        # assert v_idx.max() <= traj_length

        # [num_traj, traj_length] -> [num_traj, num_segments, num_seg_actions]
        future_returns[..., 1:] = future_q_pad_zero_end[:, q_idx]
        # for test using Q all equal 100
        # future_returns[future_returns != 0] = 100

        ################ Compute the reward in the future ##################
        # [num_traj, traj_length] -> [num_traj, traj_length + padding]
        future_r_pad_zero_end \
            = torch.nn.functional.pad(rewards, (0, num_seg_actions))

        # for test using reward all equal 1
        # future_r_pad_zero_end \
        #     = torch.nn.functional.pad(torch.ones_like(rewards), (0, num_seg_actions))

        # discount_seq as: [1, gamma, gamma^2..., 0]
        discount_idx \
            = torch.arange(traj_length + num_seg_actions, device=self.device)
        discount_seq = self.discount_factor.pow(discount_idx)
        discount_seq[-num_seg_actions:] = 0

        # Apply discount to all rewards and returns w.r.t the traj start
        # [num_traj, num_segments, 1 + traj_length]
        # TODO: check if index is set correctly
        discount_returns = future_returns * discount_seq[idx_in_segments - 1]
        # discount_returns = future_returns * discount_seq[idx_in_segments]

        # [num_traj, traj_length + 1]
        discount_r = future_r_pad_zero_end * discount_seq

        # -> [num_traj, num_segments, 1 + traj_length]
        discount_r = util.add_expand_dim(discount_r, [1],
                                         [num_segments])

        # [num_traj, num_segments, 1 + num_seg_actions]
        seg_discount_q = discount_returns

        # -> [num_traj, num_segments, 1 + num_seg_actions]
        seg_reward_idx = util.add_expand_dim(idx_in_segments, [0],
                                             [num_traj])

        # torch.gather shapes
        # input: [num_traj, num_segments, traj_length + 1]
        # index: [num_traj, num_segments, 1 + num_seg_actions]
        # result: [num_traj, num_segments, 1 + num_seg_actions]
        seg_discount_r = torch.gather(input=discount_r, dim=-1,
                                      index=seg_reward_idx)

        # [num_traj, num_segments, 1 + num_seg_actions] ->
        # [num_traj, num_segments, 1 + num_seg_actions1, 1 + num_seg_actions]
        seg_discount_r = util.add_expand_dim(seg_discount_r, [-2],
                                             [1 + num_seg_actions])

        # Get a lower triangular mask with off-diagonal elements as 1
        # [num_seg_actions + 1, num_seg_actions + 1]
        reward_tril_mask = torch.tril(torch.ones(1 + num_seg_actions,
                                                 1 + num_seg_actions,
                                                 device=self.device),
                                      diagonal=-1)

        # [num_traj, num_segments, num_seg_actions + 1, num_seg_actions + 1]
        tril_discount_rewards = seg_discount_r * reward_tril_mask

        discount_start \
            = self.discount_factor.pow(seg_start_idx)[None, :, None]

        # N-step return as target
        # V(s0) -> R0
        # Q(s0, a0) -> r0 + \gam * R1
        # Q(s0, a0, a1) -> r0 + \gam * r1 + \gam^2 * R2
        # Q(s0, a0, a1, a2) -> r0 + \gam * r1 + \gam^2 * r2 + \gam^3 * R3

        # [num_traj, num_segments, 1 + num_seg_actions]
        n_step_returns = (tril_discount_rewards.sum(
            dim=-1) + seg_discount_q) / discount_start

        ####################################################################

        return n_step_returns

    def segments_n_step_return_implicit_vf(self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            idx_in_segments: torch.Tensor,):
        """
        Segment-wise n-step return using Value function
        Use Q-func as the target of the V-func prediction (with IQL expectile loss)
        Use N-step return + V-func as the target of the Q-func prediction
        Args:
            dataset:
            idx_in_segments:

        Returns:
            n_step_returns: [num_traj, num_segments, 1 + num_seg_actions]

        """
        states = observations  # [num_traj, traj_length, dim_state]

        num_segments = idx_in_segments.shape[0]
        num_seg_actions = idx_in_segments.shape[-1] - 1
        seg_start_idx = idx_in_segments[..., 0]

        num_traj, traj_length = states.shape[0], states.shape[1]

        # NOTE: additional dimension is defined as [num_traj, num_segments]
        # TODO: MP actor related part removed, action sampling is also not necessary? use action in the dataset or directly V(s_t+n)?

        # # [num_traj, num_segments, num_seg_actions, dim_action] or
        # # [num_traj, num_segments, num_smps, num_seg_actions, dim_action]
        # actions = self.policy.sample(require_grad=False,
        #                              params_mean=params_mean_new,
        #                              params_L=params_L_new, times=action_times,
        #                              init_time=init_time,
        #                              init_pos=init_pos, init_vel=init_vel,
        #                              use_mean=False,
        #                              num_samples=self.num_samples_in_targets)
        #
        # # Normalize actions
        # if self.norm_data:
        #     actions = self.replay_buffer.normalize_data("step_actions", actions)

        ##################### Compute the current n_step Q using dataset actions  ######################
        # [num_traj, num_segments, 1 + num_seg_actions]
        future_returns = torch.zeros([num_traj, num_segments,
                                      1 + num_seg_actions], device=self.device)

        # [num_traj, dim_state] -> [num_traj, num_segments, dim_state]
        # d_state = states[
        #     torch.arange(num_traj, device=self.device), decision_idx]
        # d_state = util.add_expand_dim(d_state, [1], [num_segments])

        # [num_traj] -> [num_traj, num_segments]
        # d_idx = util.add_expand_dim(decision_idx, [1], [num_segments])

        # [num_traj, num_segments, dim_state]
        c_state = states[:, seg_start_idx]
        c_idx = util.add_expand_dim(seg_start_idx, [0], [num_traj])

        # actions in dataset
        # [num_traj, num_segments, segment_length, dim_actions]
        a_idx = idx_in_segments[:, :-1]
        d_actions = actions[:, a_idx, :]
        a_idx = util.add_expand_dim(a_idx, [0], [num_traj])

        # if self.num_samples_in_targets == 1:
        #     # [num_traj, num_segments]
        #     c_idx = util.add_expand_dim(seg_start_idx, [0], [num_traj])
        #
        #     # [num_segments, num_seg_actions]
        #     # -> [num_traj, num_segments, num_seg_actions]
        #     a_idx = util.add_expand_dim(idx_in_segments[..., :-1],
        #                                 [0], [num_traj])
        # else:
        #     num_smp = self.num_samples_in_targets
        #
        #     # -> [num_traj, num_segments, num_smp, dim_state]
        #     c_state = util.add_expand_dim(c_state, [2], [num_smp])
        #
        #     # [num_traj, num_segments, num_smp]
        #     c_idx = util.add_expand_dim(seg_start_idx, [0, 2],
        #                                 [num_traj, num_smp])
        #     # [num_segments, num_seg_actions]
        #     # -> [num_traj, num_segments, num_smp, num_seg_actions]
        #     a_idx = util.add_expand_dim(idx_in_segments[..., :-1],
        #                                 [0, 2],
        #                                 [num_traj, num_smp])

        # Use mix precision for faster computation
        with util.autocast_if(self.use_mix_precision):
            # [num_traj, num_segments, (num_smp,) 1 + num_seg_actions]
            future_q1 = self.critic.critic(self.critic.target_net1,
                                           d_state=None, c_state=c_state,
                                           actions=d_actions, idx_d=None,
                                           idx_c=c_idx, idx_a=a_idx)
            if not self.critic.single_q:
                future_q2 = self.critic.critic(self.critic.target_net2,
                                               d_state=None, c_state=c_state,
                                               actions=d_actions, idx_d=None,
                                               idx_c=c_idx, idx_a=a_idx)
            else:
                future_q2 = future_q1

        # if self.random_target and not self.targets_overestimate:
        if self.random_target:
            # Randomly choose the Q value from Q1 or Q2
            mask = torch.randint(0, 2, future_q1.shape, device=self.device)
            future_q = future_q1 * mask + future_q2 * (1 - mask)
        else:
            future_q = torch.minimum(future_q1, future_q2)

        # if self.num_samples_in_targets > 1:
        #     future_q = future_q.mean(dim=-2)

        # Use last q as the target of the V-func
        # [num_traj, num_segments, 1 + num_seg_actions]
        # -> [num_traj, num_segments]
        # Tackle the Q-func beyond the length of the trajectory
        # Option, Use the last q predictions
        future_returns[:, :-1, 0] = future_q[:, :-1, -1]

        # Find the idx where the action is the last action in the valid trajectory
        last_valid_q_idx = idx_in_segments[-1] == traj_length
        future_returns[:, -1, 0] = (
            future_q[:, -1, last_valid_q_idx].squeeze(-1))

        # # Option, Use the average q predictions, exclude the vf at 0th place
        # # Option does not work well
        # # future_returns[:, :-1, 0] = future_q[:, :-1, 1:].mean(dim=-1)
        #
        # # Find the idx where the action is the last action in the valid trajectory
        # # last_valid_q_idx = (idx_in_segments[-1] <= traj_length)[1:]
        # # future_returns[:, -1, 0] = future_q[:, -1, 1:][..., last_valid_q_idx].mean(dim=-1)

        ##################### Compute the V in the future ######################
        # state after executing the action
        # [num_traj, traj_length]
        c_idx = torch.arange(traj_length, device=self.device).long()
        c_idx = util.add_expand_dim(c_idx, [0], [num_traj])

        # [num_traj, traj_length, dim_state]
        c_state = states

        # Use mix precision for faster computation
        with util.autocast_if(self.use_mix_precision):
            # [num_traj, traj_length]
            future_v1 = self.critic.critic(self.critic.target_net1,
                                           d_state=None, c_state=c_state,
                                           actions=None, idx_d=None,
                                           idx_c=c_idx, idx_a=None).squeeze(-1)
            if not self.critic.single_q:
                future_v2 = self.critic.critic(self.critic.target_net2,
                                               d_state=None, c_state=c_state,
                                               actions=None, idx_d=None,
                                               idx_c=c_idx, idx_a=None).squeeze(-1)
            else:
                future_v2 = future_v1

        # if self.random_target and not self.targets_overestimate:
        if self.random_target:
            # Randomly choose the V value from V1 or V2
            mask = torch.randint(0, 2, future_v1.shape, device=self.device)
            future_v = future_v1 * mask + future_v2 * (1 - mask)
        else:
            future_v = torch.minimum(future_v1, future_v2)

        # Pad zeros to the states which go beyond the traj length
        future_v_pad_zero_end \
            = torch.nn.functional.pad(future_v, (0, num_seg_actions))

        # [num_segments, num_seg_actions]
        v_idx = idx_in_segments[:, 1:]
        # assert v_idx.max() <= traj_length

        # [num_traj, traj_length] -> [num_traj, num_segments, num_seg_actions]
        future_returns[..., 1:] = future_v_pad_zero_end[:, v_idx]


        ################ Compute the reward in the future ##################
        # [num_traj, traj_length] -> [num_traj, traj_length + padding]
        future_r_pad_zero_end \
            = torch.nn.functional.pad(rewards, (0, num_seg_actions))

        # discount_seq as: [1, gamma, gamma^2..., 0]
        discount_idx \
            = torch.arange(traj_length + num_seg_actions, device=self.device)
        discount_seq = self.discount_factor.pow(discount_idx)
        discount_seq[-num_seg_actions:] = 0

        # Apply discount to all rewards and returns w.r.t the traj start
        # [num_traj, num_segments, 1 + traj_length]
        discount_returns = future_returns * discount_seq[idx_in_segments]

        # [num_traj, traj_length + 1]
        discount_r = future_r_pad_zero_end * discount_seq

        # -> [num_traj, num_segments, 1 + traj_length]
        discount_r = util.add_expand_dim(discount_r, [1],
                                         [num_segments])

        # [num_traj, num_segments, 1 + num_seg_actions]
        seg_discount_q = discount_returns

        # -> [num_traj, num_segments, 1 + num_seg_actions]
        seg_reward_idx = util.add_expand_dim(idx_in_segments, [0],
                                             [num_traj])

        # torch.gather shapes
        # input: [num_traj, num_segments, traj_length + 1]
        # index: [num_traj, num_segments, 1 + num_seg_actions]
        # result: [num_traj, num_segments, 1 + num_seg_actions]
        seg_discount_r = torch.gather(input=discount_r, dim=-1,
                                      index=seg_reward_idx)

        # [num_traj, num_segments, 1 + num_seg_actions] ->
        # [num_traj, num_segments, 1 + num_seg_actions1, 1 + num_seg_actions]
        seg_discount_r = util.add_expand_dim(seg_discount_r, [-2],
                                             [1 + num_seg_actions])

        # Get a lower triangular mask with off-diagonal elements as 1
        # [num_seg_actions + 1, num_seg_actions + 1]
        reward_tril_mask = torch.tril(torch.ones(1 + num_seg_actions,
                                                 1 + num_seg_actions,
                                                 device=self.device),
                                      diagonal=-1)

        # [num_traj, num_segments, num_seg_actions + 1, num_seg_actions + 1]
        tril_discount_rewards = seg_discount_r * reward_tril_mask

        discount_start \
            = self.discount_factor.pow(seg_start_idx)[None, :, None]

        # N-step return as target
        # V(s0) -> R0
        # Q(s0, a0) -> r0 + \gam * R1
        # Q(s0, a0, a1) -> r0 + \gam * r1 + \gam^2 * R2
        # Q(s0, a0, a1, a2) -> r0 + \gam * r1 + \gam^2 * r2 + \gam^3 * R3

        # [num_traj, num_segments, 1 + num_seg_actions]
        n_step_returns = (tril_discount_rewards.sum(
            dim=-1) + seg_discount_q) / discount_start

        ####################################################################

        return n_step_returns

    def _segment_critic_update(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_observations: torch.Tensor,
            dones: torch.Tensor,
            alpha: torch.Tensor,
            log_dict: Dict,
            epochs_critic: int = 3,
            segment_length: int = 8,
            n_step_return: int = 4,
            clip_grad_norm: float = 1.0,
            use_mix_precision: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Segment-based critic update (placeholder implementation)."""
        # This is a placeholder - you can implement the full segment-based critic update here
        # For now, just use the standard Q-loss
        qf_loss, alpha_prime, alpha_prime_loss = self._q_loss(
            observations, actions, next_observations, rewards, dones, alpha, log_dict
        )
        
        return qf_loss, alpha_prime, alpha_prime_loss

    def _add_expand_dim(self, tensor: torch.Tensor, dims: List[int], sizes: List[int]) -> torch.Tensor:
        """Add expand dimensions to tensor."""
        for dim, size in zip(dims, sizes):
            tensor = tensor.unsqueeze(dim).expand(*([-1] * dim + [size] + [-1] * (tensor.dim() - dim)))
        return tensor

    def _grad_norm_clip(self, clip_norm: float, parameters) -> Tuple[float, float]:
        """Compute and clip gradient norm."""
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
        return grad_norm.item(), grad_norm.item()

    def _autocast_if(self, use_mix_precision: bool):
        """Context manager for mixed precision training."""
        if use_mix_precision:
            return torch.cuda.amp.autocast()
        else:
            from contextlib import nullcontext
            return nullcontext()


    def train(self, batch: TensorBatch) -> Dict[str, float]:
        torch.autograd.set_detect_anomaly(True)
        mse = torch.nn.MSELoss()

        step_batch = [item[:, 0] for item in batch]
        (
            observations_step,
            actions_step,
            rewards_step,
            next_observations_step,
            dones_step,
        ) = step_batch
        self.total_it += 1

        # step_batch = [item[:, 0] for item in batch]
        #
        # (
        #     observations_seq,
        #     actions_seq,
        #     rewards_seq,
        #     next_observations_seq,
        #     dones_seq,
        # ) = batch
        #
        # rewards_seq = rewards_seq.squeeze()  # [num_traj, traj_length]
        # # actions_seq = rewards_seq.squeeze() # [num_traj, traj_length, dim_action]
        #
        # (
        #     observations_step,
        #     actions_step,
        #     rewards_step,
        #     next_observations_step,
        #     dones_step,
        # ) = step_batch
        # self.total_it += 1

        new_actions, log_pi = self.actor(observations_step)
        # new_actions, log_pi = self.actor.sample(observations_step, self.num_samples_in_policy)
        # new_actions_seq = torch.cat([new_actions, actions_seq[..., 1:]], dim=-1,)
        # TODO: use actions_seq to feed transformer to get Q(s, a)
        if self.num_samples_in_policy == 1:
            new_actions = new_actions.squeeze()

        alpha, alpha_loss = self._alpha_and_alpha_loss(observations_step, log_pi)

        # TODO: put back after segment critic update is implemented
        """ Policy loss """
        # policy_loss = self._policy_loss(
        #     observations_step, actions_step, new_actions, alpha, log_pi
        # )
        #
        policy_loss = self._policy_loss(
            observations_step, actions_step, new_actions, alpha, log_pi
        )

        log_dict = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
        )

        """ Q function loss """
        ########################################################################
        #                             Update critic                            #
        ########################################################################
        # Check if critic is SeqCritic
        # if self.use_segment_critic_update or self.use_segment_n_step_return_qf:
        if isinstance(self.critic, seq_critic.SeqCritic):
            util.run_time_test(lock=True, key="update critic")
            self.critic.train()
            self.critic.requires_grad(True)
            # Initialize scalers for mixed precision
            if self.use_mix_precision:
                scaler_1 = torch.cuda.amp.GradScaler()
                scaler_2 = torch.cuda.amp.GradScaler()
            else:
                scaler_1 = scaler_2 = None

            # Use segment-based n-step return Q-learning
            critic_loss_list = []
            critic_grad_norm = []
            clipped_critic_grad_norm = []

            # Time logs for critic updates
            update_critic_net_time = []
            update_target_net_time = []

            # Value estimation error log
            mc_returns_list = []
            targets_list = []
            targets_bias_list = []

            # Generate segments using the reusable get_segments method
            num_traj = observations_seq.shape[0]
            traj_length = observations_seq.shape[1]

            # Set traj_length for get_segments method
            self.traj_length = traj_length

            if self.return_type == "segment_n_step_return_vf":
                raise NotImplementedError
                # # Use segment-based n-step return Q-learning (similar to SeqQAgent)
                # idx_in_segments = self.get_segments(pad_additional=True)
                # seg_start_idx = idx_in_segments[..., 0]
                # assert seg_start_idx[-1] < self.traj_length
                # seg_actions_idx = idx_in_segments[..., :-1]
                # num_seg_actions = seg_actions_idx.shape[-1]
                #
                # # [num_traj, num_segments, dim_state]
                # c_state = observations_seq[:, seg_start_idx]
                # seg_start_idx = util.add_expand_dim(seg_start_idx, [0], [num_traj])
                #
                # # for CQL step-based actor, c_state_seq [num_traj, num_segments, segments_length, dim_state]
                # seg_state_idx = idx_in_segments[..., :-1]
                # num_seg_state = seg_state_idx.shape[-1]
                # padded_states_c = torch.nn.functional.pad(
                #     observations_seq, (0, 0, 0, num_seg_state), "constant", 0)
                # padded_states_n = torch.nn.functional.pad(
                #     next_observations_seq, (0, 0, 0, num_seg_state), "constant", 0)
                #
                # c_state_seq = padded_states_c[:, seg_state_idx]
                # n_state_seq = padded_states_n[:, seg_state_idx]
                # seg_state_idx = util.add_expand_dim(seg_state_idx, [0], [num_traj])
                #
                # padded_actions = torch.nn.functional.pad(
                #     actions_seq, (0, 0, 0, num_seg_actions), "constant", 0)
                #
                # # [num_traj, num_segments, num_seg_actions, dim_action]
                # seg_actions = padded_actions[:, seg_actions_idx]
                #
                # # [num_traj, num_segments, num_seg_actions]
                # seg_actions_idx = util.add_expand_dim(seg_actions_idx, [0],
                #                                       [num_traj])
                #
                # targets = self.segments_n_step_return_vf(
                #     observations_seq,
                #     actions_seq,
                #     rewards_seq,
                #     dones_seq,
                #     idx_in_segments)
                #
                # # Log targets and MC returns
                # # if self.log_now:
                # mc_returns_mean = util.compute_mc_return(
                #     rewards_seq.mean(dim=0),
                #     self.discount_factor).mean().item()
                #
                # mc_returns_list.append(mc_returns_mean)
                # targets_mean = targets.mean().item()
                # targets_list.append(targets_mean)
                # targets_bias_list.append(targets_mean - mc_returns_mean)
                # log_dict.update(
                #     dict(
                #         mc_returns_mean=mc_returns_mean.item(),
                #         targets_mean=targets_mean.item(),
                #         targets_bias_mean=(targets_mean - mc_returns_mean).item(),
                #     )
                # )
                #
                # for net, target_net, opt, scaler in self.critic_nets_and_opt():
                #     # Use mix precision for faster computation
                #     with util.autocast_if(self.use_mix_precision):
                #         # [num_traj, num_segments, 1 + num_seg_actions]
                #         vq_pred = self.critic.critic(
                #             net=net, d_state=None, c_state=c_state,
                #             actions=seg_actions, idx_d=None, idx_c=seg_start_idx,
                #             idx_a=seg_actions_idx)
                #
                #         # Mask out the padded actions
                #         # [num_traj, num_segments, num_seg_actions]
                #         valid_mask = seg_actions_idx < self.traj_length
                #
                #         # [num_traj, num_segments, num_seg_actions]
                #         vq_pred[..., 1:] = vq_pred[..., 1:] * valid_mask
                #         targets[..., 1:] = targets[..., 1:] * valid_mask
                #
                #         # Loss
                #         vq_diff = vq_pred[..., 1:] - targets[..., 1:]
                #         vq_sum = torch.sum(vq_diff ** 2)
                #         critic_loss = mse(vq_pred, targets)  # Both V and Q
                #
                #         # print("critic_loss", critic_loss.item())
                #
                #         cql_loss, alpha_prime, alpha_prime_loss = self.cql_critic_loss(net=net,
                #                                                                        c_state=c_state,
                #                                                                        c_state_seq=c_state_seq,
                #                                                                        n_state_seq=n_state_seq,
                #                                                                        seg_actions=seg_actions,
                #                                                                        seg_start_idx=seg_start_idx,
                #                                                                        seg_actions_idx=seg_actions_idx,
                #                                                                        seg_state_idx=seg_state_idx,
                #                                                                        traj_length=traj_length,
                #                                                                        targets=targets,
                #                                                                        vq_predict=vq_pred,
                #                                                                        )
                #
                #     # Update critic net parameters
                #     util.run_time_test(lock=True, key="update critic net")
                #     opt.zero_grad(set_to_none=True)
                #
                #     # critic_loss.backward()
                #     scaler.scale(critic_loss).backward()
                #
                #     if self.clip_grad_norm > 0 or self.log_now:
                #         grad_norm, grad_norm_c = util.grad_norm_clip(
                #             self.clip_grad_norm, net.parameters())
                #     else:
                #         grad_norm, grad_norm_c = 0., 0.
                #
                #     # opt.step()
                #     scaler.step(opt)
                #     scaler.update()
                #
                #     update_critic_net_time.append(
                #         util.run_time_test(lock=False,
                #                            key="update critic net"))
                #
                #     # Logging
                #     critic_loss_list.append(critic_loss.item())
                #     critic_grad_norm.append(grad_norm)
                #     clipped_critic_grad_norm.append(grad_norm_c)
                #
                #     # Update target network
                #     util.run_time_test(lock=True, key="copy critic net")
                #     self.critic.update_target_net(net, target_net)
                #     update_target_net_time.append(
                #         util.run_time_test(lock=False,
                #                            key="copy critic net"))
            elif self.return_type == "segment_n_step_return_qf":
                # Use segment-based n-step return Q-learning (similar to SeqQAgent)
                idx_in_segments = self.get_segments(pad_additional=True)
                seg_start_idx = idx_in_segments[..., 0]
                assert seg_start_idx[-1] < self.traj_length
                seg_actions_idx = idx_in_segments[..., :-1]
                num_seg_actions = seg_actions_idx.shape[-1]

                # [num_traj, num_segments, dim_state]
                c_state = observations_seq[:, seg_start_idx]
                seg_start_idx = util.add_expand_dim(seg_start_idx, [0], [num_traj])

                # for CQL step-based actor, c_state_seq [num_traj, num_segments, segments_length, dim_state]
                seg_state_idx = idx_in_segments[..., :-1]
                num_seg_state = seg_state_idx.shape[-1]
                padded_states_c = torch.nn.functional.pad(
                    observations_seq, (0, 0, 0, num_seg_state), "constant", 0)
                padded_states_n = torch.nn.functional.pad(
                    next_observations_seq, (0, 0, 0, num_seg_state), "constant", 0)

                c_state_seq = padded_states_c[:, seg_state_idx]
                n_state_seq = padded_states_n[:, seg_state_idx]
                seg_state_idx = util.add_expand_dim(seg_state_idx, [0], [num_traj])

                padded_actions = torch.nn.functional.pad(
                    actions_seq, (0, 0, 0, num_seg_actions), "constant", 0)

                # [num_traj, num_segments, num_seg_actions, dim_action]
                seg_actions = padded_actions[:, seg_actions_idx]

                # [num_traj, num_segments, num_seg_actions]
                seg_actions_idx = util.add_expand_dim(seg_actions_idx, [0],
                                                      [num_traj])

                targets = self.segments_n_step_return_qf(
                    observations_seq,
                    actions_seq,
                    next_observations_seq,
                    rewards_seq,
                    dones_seq,
                    idx_in_segments)

                # Log targets and MC returns
                # if self.log_now:
                mc_returns_mean = util.compute_mc_return(
                    rewards_seq.mean(dim=0),
                    self.discount_factor).mean().item()

                mc_returns_list.append(mc_returns_mean)
                targets_mean = targets.mean().item()
                targets_list.append(targets_mean)
                targets_bias_list.append(targets_mean - mc_returns_mean)

                log_dict.update(
                    dict(
                        mc_returns_mean=mc_returns_mean,
                        targets_mean=targets_mean,
                        targets_bias_mean=targets_mean - mc_returns_mean,
                    )
                )

                # qf_loss = 0.0

                for net_name, net, target_net, opt, scaler in self.critic_nets_and_opt():
                    # Use mix precision for faster computation
                    with util.autocast_if(self.use_mix_precision):
                        # [num_traj, num_segments, 1 + num_seg_actions]
                        vq_pred = self.critic.critic(
                            net=net, d_state=None, c_state=c_state,
                            actions=seg_actions, idx_d=None, idx_c=seg_start_idx,
                            idx_a=seg_actions_idx)

                        # Mask out the padded actions
                        # [num_traj, num_segments, num_seg_actions]
                        valid_mask = seg_actions_idx < self.traj_length

                        # [num_traj, num_segments, num_seg_actions]
                        vq_pred[..., 1:] = vq_pred[..., 1:] * valid_mask
                        targets[..., 1:] = targets[..., 1:] * valid_mask

                        # Loss
                        vq_diff = vq_pred[..., 1:] - targets[..., 1:]
                        vq_sum = torch.sum(vq_diff ** 2)
                        critic_loss = mse(vq_pred[..., 1:] , targets[..., 1:])  # Both V and Q

                        # print("critic_loss", critic_loss.item())

                        cql_loss, alpha_prime, alpha_prime_loss = self.cql_critic_loss(net=net,
                                                                                       c_state=c_state,
                                                                                       c_state_seq=c_state_seq,
                                                                                       n_state_seq=n_state_seq,
                                                                                       seg_actions=seg_actions,
                                                                                       seg_start_idx=seg_start_idx,
                                                                                       seg_actions_idx=seg_actions_idx,
                                                                                       seg_state_idx=seg_state_idx,
                                                                                       traj_length=traj_length,
                                                                                       targets=targets,
                                                                                       vq_predict=vq_pred,
                                                                                       )

                        qf1_value = vq_pred[..., 1] - targets[..., 1] # log Q(s, a)

                        log_dict.update(
                            {
                                f"{net_name}_1step_q_diff": qf1_value.mean().item(),
                                f"{net_name}_1step_q": vq_pred[..., 1].mean().item(),
                                f"{net_name}_q_mean": vq_pred[..., 1:].mean().item(),
                                f"{net_name}_target_mean": targets[..., 1:].mean().item(),
                                f"{net_name}_return_mean": mc_returns_mean,
                                f"{net_name}_v": vq_pred[..., 0].mean().item(),
                                f"{net_name}_q_loss": critic_loss.item(),
                                f"{net_name}_cql_loss":cql_loss.item(),
                            }
                        )

                        # total loss for this critic
                        critic_loss = critic_loss + cql_loss

                        # # accumulate into qf_loss
                        # qf_loss = qf_loss + loss

                        # Update critic net parameters
                    util.run_time_test(lock=True, key="update critic net")
                    opt.zero_grad(set_to_none=True)

                    # critic_loss.backward()
                    scaler.scale(critic_loss).backward()

                    if self.clip_grad_norm > 0 or self.log_now:
                        grad_norm, grad_norm_c = util.grad_norm_clip(
                            self.clip_grad_norm, net.parameters())
                    else:
                        grad_norm, grad_norm_c = 0., 0.

                    # opt.step()
                    scaler.step(opt)
                    scaler.update()

                    if self.cql_lagrange:
                        self.alpha_prime_optimizer.zero_grad()
                        alpha_prime_loss.backward()
                        self.alpha_prime_optimizer.step()

                    update_critic_net_time.append(
                        util.run_time_test(lock=False,
                                           key="update critic net"))

                    # Logging
                    critic_loss_list.append(critic_loss.item())
                    critic_grad_norm.append(grad_norm)
                    clipped_critic_grad_norm.append(grad_norm_c)

                    # Update target network
                    util.run_time_test(lock=True, key="copy critic net")
                    self.critic.update_target_net(net, target_net)
                    update_target_net_time.append(
                        util.run_time_test(lock=False,
                                           key="copy critic net"))
            elif self.return_type == "segment_n_step_return_implicit_vf":
                # Use segment-based n-step return Q-learning (similar to SeqQAgent)
                idx_in_segments = self.get_segments(pad_additional=True)
                seg_start_idx = idx_in_segments[..., 0]
                assert seg_start_idx[-1] < self.traj_length
                seg_actions_idx = idx_in_segments[..., :-1]
                num_seg_actions = seg_actions_idx.shape[-1]

                # [num_traj, num_segments, dim_state]
                c_state = observations_seq[:, seg_start_idx]
                seg_start_idx = util.add_expand_dim(seg_start_idx, [0], [num_traj])

                # for CQL step-based actor, c_state_seq [num_traj, num_segments, segments_length, dim_state]
                seg_state_idx = idx_in_segments[..., :-1]
                num_seg_state = seg_state_idx.shape[-1]
                padded_states_c = torch.nn.functional.pad(
                    observations_seq, (0, 0, 0, num_seg_state), "constant", 0)
                padded_states_n = torch.nn.functional.pad(
                    next_observations_seq, (0, 0, 0, num_seg_state), "constant", 0)

                c_state_seq = padded_states_c[:, seg_state_idx]
                n_state_seq = padded_states_n[:, seg_state_idx]
                seg_state_idx = util.add_expand_dim(seg_state_idx, [0], [num_traj])

                padded_actions = torch.nn.functional.pad(
                    actions_seq, (0, 0, 0, num_seg_actions), "constant", 0)

                # [num_traj, num_segments, num_seg_actions, dim_action]
                seg_actions = padded_actions[:, seg_actions_idx]

                # [num_traj, num_segments, num_seg_actions]
                seg_actions_idx = util.add_expand_dim(seg_actions_idx, [0],
                                                      [num_traj])

                targets = self.segments_n_step_return_implicit_vf(
                    observations_seq,
                    actions_seq,
                    rewards_seq,
                    dones_seq,
                    idx_in_segments)

                # Log targets and MC returns
                # if self.log_now:
                mc_returns_mean = util.compute_mc_return(
                    rewards_seq.mean(dim=0),
                    self.discount_factor).mean().item()

                mc_returns_list.append(mc_returns_mean)
                targets_mean = targets.mean().item()
                targets_list.append(targets_mean)
                targets_bias_list.append(targets_mean - mc_returns_mean)

                log_dict.update(
                    dict(
                        mc_returns_mean=mc_returns_mean,
                        targets_mean=targets_mean,
                        targets_bias_mean=targets_mean - mc_returns_mean,
                    )
                )

                # qf_loss = 0.0

                for net_name, net, target_net, opt, scaler in self.critic_nets_and_opt():
                    # Use mix precision for faster computation
                    with util.autocast_if(self.use_mix_precision):
                        # [num_traj, num_segments, 1 + num_seg_actions]
                        vq_pred = self.critic.critic(
                            net=net, d_state=None, c_state=c_state,
                            actions=seg_actions, idx_d=None, idx_c=seg_start_idx,
                            idx_a=seg_actions_idx)

                        # Mask out the padded actions
                        # [num_traj, num_segments, num_seg_actions]
                        valid_mask = seg_actions_idx < self.traj_length

                        # [num_traj, num_segments, num_seg_actions]
                        vq_pred[..., 1:] = vq_pred[..., 1:] * valid_mask
                        targets[..., 1:] = targets[..., 1:] * valid_mask

                        # Loss
                        q_diff = vq_pred[..., 1:] - targets[..., 1:]
                        q_sum = torch.sum(q_diff ** 2)
                        v_diff = vq_pred[..., 0] - targets[..., 0]
                        v_sum = torch.sum(v_diff ** 2)
                        critic_loss = mse(vq_pred[..., 1:], targets[..., 1:])
                        # print(vq_pred[..., 1:], targets[..., 1:], critic_loss, ((vq_pred[..., 1:]-targets[..., 1:])**2).mean())
                        iql_loss = self.asymmetric_l2_loss(v_diff, self.iql_tau)
                        critic_loss = critic_loss + iql_loss
                        # expectile loss for  V and mse for Q

                        qf1_value = vq_pred[..., 1] - targets[..., 1]

                        # qf_loss = qf_loss + critic_loss
                        # print("critic_loss", critic_loss.item())

                        log_dict.update(
                            {
                                f"{net_name}_1step_q_diff": qf1_value.mean().item(),
                                f"{net_name}_1step_q": vq_pred[..., 1].mean().item(),
                                f"{net_name}_q_mean": vq_pred[..., 1:].mean().item(),
                                f"{net_name}_target_mean": targets[..., 1:].mean().item(),
                                f"{net_name}_return_mean": mc_returns_mean,
                                f"{net_name}_v": vq_pred[..., 0].mean().item(),
                                f"{net_name}_q_loss": critic_loss.item(),
                                f"{net_name}_iql_loss": iql_loss.item(),
                            }
                        )

                    # Update critic net parameters
                    util.run_time_test(lock=True, key="update critic net")
                    opt.zero_grad(set_to_none=True)

                    # critic_loss.backward()
                    scaler.scale(critic_loss).backward()

                    if self.clip_grad_norm > 0 or self.log_now:
                        grad_norm, grad_norm_c = util.grad_norm_clip(
                            self.clip_grad_norm, net.parameters())
                    else:
                        grad_norm, grad_norm_c = 0., 0.

                    # opt.step()
                    scaler.step(opt)
                    scaler.update()

                    update_critic_net_time.append(
                        util.run_time_test(lock=False,
                                           key="update critic net"))

                    # Logging
                    critic_loss_list.append(critic_loss.item())
                    critic_grad_norm.append(grad_norm)
                    clipped_critic_grad_norm.append(grad_norm_c)

                    # Update target network
                    util.run_time_test(lock=True, key="copy critic net")
                    self.critic.update_target_net(net, target_net)
                    update_target_net_time.append(
                        util.run_time_test(lock=False,
                                           key="copy critic net"))
            # Note: Use N-step V-func return for segment-wise update
            elif self.return_type == "v_func":
                raise NotImplementedError

            # Note: Use true return for update
            elif self.return_type == "true_return":
                raise NotImplementedError
            else:
                raise ValueError("Unknown return type")

            update_critic_time = util.run_time_test(lock=False, key="update critic")
            self.targets_overestimate = np.mean(targets_bias_list) > 1
        else:
            # Use original critic update
            qf_loss, alpha_prime, alpha_prime_loss = self._q_loss(
                observations_step,
                actions_step,
                next_observations_step,
                rewards_step,
                dones_step,
                alpha,
                log_dict
            )
        # ########################################################################
        # #                             Update policy
        # ########################################################################
        # util.run_time_test(lock=True, key="update policy")
        #
        # for grad_idx in range(self.epochs_policy):
        #     if not update_policy:
        #         break

        # # freeze critic params for the actor update
        # for p in self.critic.parameters:
        #     for cp in p:
        #         cp.requires_grad_(False)

        # policy_loss = self._policy_loss(
        #     observations_step, actions_step, new_actions, alpha, log_pi
        # )

        # log_dict.update(
        #     dict(
        #     policy_loss=policy_loss.item(),
        #     )
        # )

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # # unfreeze critic params afterwards
        # for p in self.critic.parameters:
        #     for cp in p:
        #         cp.requires_grad_(True)

        # Only update critics if not using segment-based update (already done in _segment_critic_update or _segment_n_step_return_qf)
        # if not (self.use_segment_critic_update or self.use_segment_n_step_return_qf):
        # self.critic_1_optimizer.zero_grad(set_to_none=True)
        # self.critic_2_optimizer.zero_grad(set_to_none=True)
        # # qf_loss.backward(retain_graph=True)
        #
        # qf_loss.backward()
        #
        # self.critic_1_optimizer.step()
        # self.critic_2_optimizer.step()

        # for cql_test
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "sac_log_alpha": self.log_alpha,
            "sac_log_alpha_optim": self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
            "cql_log_alpha": self.log_alpha_prime,
            "cql_log_alpha_optim": self.alpha_prime_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.critic.load_state_dict(state_dict=state_dict["critic"])

        self.critic_1_optimizer.load_state_dict(
            state_dict=state_dict["critic_1_optimizer"]
        )
        self.critic_2_optimizer.load_state_dict(
            state_dict=state_dict["critic_2_optimizer"]
        )
        self.actor_optimizer.load_state_dict(state_dict=state_dict["actor_optim"])

        if state_dict["sac_log_alpha"] is not None:
            self.log_alpha = state_dict["sac_log_alpha"]
            if state_dict["sac_log_alpha_optim"] is not None:
                self.alpha_optimizer.load_state_dict(
                    state_dict=state_dict["sac_log_alpha_optim"]
                )

        self.log_alpha_prime = state_dict["cql_log_alpha"]
        self.alpha_prime_optimizer.load_state_dict(
            state_dict=state_dict["cql_log_alpha_optim"]
        )
        self.total_it = state_dict["total_it"]

    def critic_nets_and_opt(self):
        if self.critic.single_q:
            return zip(["net1"],
                       util.make_iterable(self.critic.net1, "list"),
                       util.make_iterable(self.critic.target_net1, "list"),
                       util.make_iterable(self.critic_optimizer[0], "list"),
                       util.make_iterable(self.critic_grad_scaler[0], "list"))
        else:
            return zip(["net1", "net2"],
                       [self.critic.net1, self.critic.net2],
                       [self.critic.target_net1, self.critic.target_net2],
                       self.critic_optimizer, self.critic_grad_scaler)

    def cql_critic_loss(
            self,
            # critic,  # your wrapper with .critic(net=..., c_state=..., actions=..., idx_c=..., idx_a=...)
            net,  # the specific critic net to train (e.g., net1 or net2)
            # actor,  # step-based actor; actor(observations, repeat=K) -> (actions, log_pi)
            c_state,  # [B, T, d_s] (or whatever your critic expects)
            c_state_seq, # [B, S, L, d_s] (segment states)
            n_state_seq,  # [B, S, L, d_s] next states for segments
            seg_actions,  # [B, S, L, d_a] actions along each segment (data actions)
            seg_start_idx,  # [B, S] start indices t per segment
            seg_actions_idx,  # [B, S, L] absolute indices of a_t...a_{t+L-1}
            seg_state_idx,  # [B, S, L] absolute indices of s_t...s_{t+L-1}
            traj_length: int,  # T
            targets,  # [B, S, 1+L] N-step targets (V + L Qs) — should be detached upstream
            vq_predict,  # [B, S, 1+L] Q-values predicted by the critic for the actions in dataset
            # # --- CQL knobs ---
            # cql_n_actions: int = 10,
            # cql_alpha: float = 10.0,
            # cql_temp: float = 1.0,
            # cql_importance_sample: bool = True,
            # cql_use_lagrange: bool = False,
            # log_alpha_prime: torch.nn.Module = None,  # Scalar() module if using Lagrange
            # cql_target_action_gap: float = -1.0,
            # random action range
            act_low: float = -1.0,
            act_high: float = 1.0,
    ):
        """
            Returns:
                cql_loss: scalar (conservative penalty)
                stats: dict of scalars for logging
        """

        # [num_batch, num_segments, num_seg_actions, dim_action]
        B, S, L, d_a = seg_actions.shape
        device = seg_actions.device
        dtype = seg_actions.dtype

        # generate random actions along the segments
        # [num_batch, num_segments, num_smps, num_seg_actions, dim_action]
        a_rand = torch.empty((B, S, self.num_samples_in_cql_loss, L, d_a), dtype=dtype, device=device).uniform_(act_low, act_high)
        range_len = act_high - act_low
        random_density = torch.log((1.0/range_len)**d_a * torch.ones((B, S, self.num_samples_in_cql_loss, L), device=device))

        # # generate t and t+1 step-based actions using current states and next states
        # # [num_batch, num_segments, num_smps, num_seg_actions, dim_action]
        a_step_c, log_pi_c = self.actor.sample(c_state_seq, self.num_samples_in_cql_loss)
        a_step_n, log_pi_n = self.actor.sample(n_state_seq, self.num_samples_in_cql_loss)

        # a_step_c, log_pi_c = self.actor(c_state_seq)
        # a_step_n, log_pi_n = self.actor(n_state_seq)
        # TODO: add repeat for top-k sampling

        # c_state = util.add_expand_dim(c_state, [2], [self.num_samples_in_cql_loss])
        if self.num_samples_in_cql_loss == 1:
            # [num_traj, num_segments]
            c_idx = util.add_expand_dim(seg_start_idx, [0], [B])

            # [num_segments, num_seg_actions]
            # -> [num_traj, num_segments, num_seg_actions]
            a_idx = seg_actions_idx
        else:
            num_smp = self.num_samples_in_cql_loss

            # -> [num_traj, num_segments, num_smp, dim_state]
            c_state = util.add_expand_dim(c_state, [2], [num_smp])

            # [num_traj, num_segments, num_smp]
            c_idx = util.add_expand_dim(seg_start_idx, [2],
                                        [num_smp])
            # [num_segments, num_seg_actions]
            # -> [num_traj, num_segments, num_smp, num_seg_actions]
            a_idx = util.add_expand_dim(seg_actions_idx,
                                        [2],
                                        [num_smp])


        cql_rand = self.critic.critic(
            net=net, d_state=None, c_state=c_state,
            actions=a_rand, idx_d=None, idx_c=c_idx,
            idx_a=a_idx)
        cql_current_actions = self.critic.critic(
            net=net, d_state=None, c_state=c_state,
            actions=a_step_c, idx_d=None, idx_c=c_idx,
            idx_a=a_idx)
        cql_next_actions = self.critic.critic(
            net=net, d_state=None, c_state=c_state,
            actions=a_step_n, idx_d=None, idx_c=c_idx,
            idx_a=a_idx)

        # TODO: cat dim incorrect, implement top-k sampling first
        # [num_batch, num_segments, segment_length] -> [num_batch, num_segments, segment_length, 1 + 3*K_samples]
        if self.num_samples_in_cql_loss == 1:
            cql_cat = torch.cat(
                [
                    cql_rand[..., 1:].unsqueeze(-2),
                    vq_predict[..., 1:].unsqueeze(-2),
                    cql_next_actions[..., 1:].unsqueeze(-2),
                    cql_current_actions[..., 1:].unsqueeze(-2),
                ],
                dim=-2,
            )
        else:
            cql_cat = torch.cat(
                [
                    cql_rand[..., 1:],
                    vq_predict[..., 1:].unsqueeze(-2),
                    cql_next_actions[..., 1:],
                    cql_current_actions[..., 1:],
                ],
                dim=-2,
            )
        # TODO: should vq_predict be included here?
        cql_std = torch.std(cql_cat, dim=-2)

        # Importance sampling weight for CQL
        if self.cql_importance_sample:
            cql_cat = torch.cat(
                [
                    cql_rand[..., 1:] - random_density,
                    cql_next_actions[..., 1:] - log_pi_n.detach(),
                    cql_current_actions[..., 1:] - log_pi_c.detach(),
                ],
                dim=-2,
            )

        cql_ood = torch.logsumexp(cql_cat / self.cql_temp, dim=-2) * self.cql_temp

        # Subtract the log likelihood of data
        cql_diff = torch.clamp(
            cql_ood - vq_predict[..., 1:],
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
            )
            cql_min_loss = (
                alpha_prime
                * self.cql_alpha
                * (cql_diff - self.cql_target_action_gap)
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_loss) * 0.5
            # alpha_prime_loss.backward(retain_graph=True)
            # self.alpha_prime_optimizer.step()
        else:
            cql_min_loss = cql_diff * self.cql_alpha
            alpha_prime_loss = seg_actions.new_tensor(0.0)
            alpha_prime = seg_actions.new_tensor(0.0)

        cql_loss = cql_min_loss

        # print("cql_loss", cql_loss.item())

        return cql_loss, alpha_prime, alpha_prime_loss

    def asymmetric_l2_loss(self, u: torch.Tensor, tau: float) -> torch.Tensor:
        return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


@pyrallis.wrap()
def train(config: TrainConfig, cw_config: dict = None) -> None:
    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # dataset = d4rl.qlearning_dataset(env)
    dataset_2 = d4rl.qlearning_dataset(env, terminate_on_end=True)

    if config.normalize_reward:
        # modify_reward(
        #     dataset,
        #     config.env,
        #     reward_scale=config.reward_scale,
        #     reward_bias=config.reward_bias,
        # )

        modify_reward(
            dataset_2,
            config.env,
            reward_scale=config.reward_scale,
            reward_bias=config.reward_bias,
        )

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset_2["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset_2["observations"] = normalize_states(
        dataset_2["observations"], state_mean, state_std
    )
    dataset_2["next_observations"] = normalize_states(
        dataset_2["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    # replay_buffer = ReplayBuffer(
    #     state_dim,
    #     action_dim,
    #     config.buffer_size,
    #     config.device,
    # )
    # replay_buffer.load_d4rl_dataset(dataset_2)

    replay_buffer_data_shape = {
        "observations": (state_dim,),
        "actions": (action_dim,),
        "rewards": (1,),
        "next_observations": (state_dim,),
        "terminals": (1,),
    }
    replay_buffer_norm_info = {
        "observations": True,
        "actions": False,
        "rewards": False,
        "next_observations": True,
        "terminals": False,
    }

    # replay_buffer_modular = rb.StepReplayBuffer(
    #     replay_buffer_data_shape,
    #     replay_buffer_norm_info,
    #     config.buffer_size,
    #     device=config.device,
    # )
    #
    # replay_buffer_modular.load_d4rl_dataset(dataset_2)
    # replay_buffer_modular.update_buffer_normalizer()


    replay_buffer_modular_seq = rb.SeqReplayBuffer(
        replay_buffer_data_shape,
        replay_buffer_norm_info,
        config.buffer_size,
        device=config.device,
    )

    replay_buffer_modular_seq.load_d4rl_dataset(dataset_2)
    replay_buffer_modular_seq.update_buffer_normalizer()

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    # critic_config = {
    #     "state_dim": state_dim,
    #     "action_dim": action_dim,
    #     "single_q": False,
    #     "device": config.device,
    #     "dtype": "float32",
    #     "update_rate": config.soft_target_update_rate,
    #     "orthogonal_init": config.orthogonal_init,
    #     "q_n_hidden_layers": config.q_n_hidden_layers,
    #     "cql_alpha": config.cql_alpha,
    #     "cql_temp": config.cql_temp,
    #     "cql_n_actions": config.cql_n_actions,
    #     "discount": config.discount,
    #     "backup_entropy": config.backup_entropy,
    # }
    #
    # critic = crit.CQLCritic(**critic_config)

    critic_config = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "single_q": False,
        "device": config.device,
        "dtype": "float32",
        "update_rate": config.soft_target_update_rate,
        "bias": True,
        "n_embd": 128,
        "block_size": 1024,
        "dropout": 0.0,
        "n_layer": 2,
        "n_head": 8,
        "update_rate": 0.005,
        "use_layer_norm": True,
        "relative_pos": False
    }

    # critic = seq_critic.SeqCritic(**critic_config)

    # critic_1, critic_2 = critic.net1, critic.net2
    # target_critic_1, target_critic_2 = critic.target_net1, critic.target_net2
    # critic_1_optimizer, critic_2_optimizer = critic.configure_optimizer(
    #     weight_decay=0.0,
    #     learning_rate=config.qf_lr,
    #     betas=(0.9, 0.999),
    # )

    # CQL original codebase test
    cql_critic_1 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)
    cql_critic_2 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
    ).to(config.device)

    cql_critic_1_optimizer = torch.optim.Adam(list(cql_critic_1.parameters()), config.qf_lr)
    cql_critic_2_optimizer = torch.optim.Adam(list(cql_critic_2.parameters()), config.qf_lr)
    # cql_target_critic_1 = deepcopy(cql_critic_1).to(config.device)
    # cql_target_critic_2 = deepcopy(cql_critic_2).to(config.device)

    cql_actor = TanhGaussianPolicy(
        state_dim,
        action_dim,
        max_action,
        log_std_multiplier=config.policy_log_std_multiplier,
        orthogonal_init=config.orthogonal_init,
    ).to(config.device)
    cql_actor_optimizer = torch.optim.Adam(cql_actor.parameters(), config.policy_lr)

    cql_kwargs = {
        "critic_1": cql_critic_1,
        "critic_2": cql_critic_2,
        "critic_1_optimizer": cql_critic_1_optimizer,
        "critic_2_optimizer": cql_critic_2_optimizer,
        "actor": cql_actor,
        "actor_optimizer": cql_actor_optimizer,
        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        # CQL
        "target_entropy": -np.prod(env.action_space.shape).item(),
        "alpha_multiplier": config.alpha_multiplier,
        "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
        "backup_entropy": config.backup_entropy,
        "policy_lr": config.policy_lr,
        "qf_lr": config.qf_lr,
        "bc_steps": config.bc_steps,
        "target_update_period": config.target_update_period,
        "cql_n_actions": config.cql_n_actions,
        "cql_importance_sample": config.cql_importance_sample,
        "cql_lagrange": config.cql_lagrange,
        "cql_target_action_gap": config.cql_target_action_gap,
        "cql_temp": config.cql_temp,
        "cql_alpha": config.cql_alpha,
        "cql_max_target_backup": config.cql_max_target_backup,
        "cql_clip_diff_min": config.cql_clip_diff_min,
        "cql_clip_diff_max": config.cql_clip_diff_max,
    }


    # policy_kwargs = {
    #     "mean_net_args": {
    #         "avg_neuron": 256,
    #         "num_hidden": 3,
    #         "shape": 0.0,
    #     },
    #     "variance_net_args": {
    #         "std_only": True,
    #         "contextual": True,
    #         "avg_neuron": 256,
    #         "num_hidden": 3,
    #         "shape": 0.0,
    #     },
    #     "init_method": "orthogonal",
    #     "out_layer_gain": 0.01,
    #     "min_std": 1e-5,
    #     "act_func_hidden": "leaky_relu",
    #     "act_func_last": None,
    # }

    # actor = pl.TanhGaussianPolicy(
    #     state_dim,
    #     action_dim,
    #     max_action=max_action,
    #     log_std_multiplier=config.policy_log_std_multiplier,
    #     orthogonal_init=config.orthogonal_init,
    #     device=config.device,
    #     **policy_kwargs,
    # )
    #
    # actor_optimizer = torch.optim.Adam(actor.parameters, config.policy_lr)

    # kwargs = {
    #     # "critic_1": critic_1,
    #     # "critic_2": critic_2,
    #     # "target_critic_1": target_critic_1,
    #     # "target_critic_2": target_critic_2,
    #     # "critic_1_optimizer": critic_1_optimizer,
    #     # "critic_2_optimizer": critic_2_optimizer,
    #     "critic": critic,
    #     "actor": cql_actor,
    #     "actor_optimizer": actor_optimizer,
    #     "discount": config.discount,
    #     "soft_target_update_rate": config.soft_target_update_rate,
    #     "device": config.device,
    #     # CQL
    #     "target_entropy": -np.prod(env.action_space.shape).item(),
    #     "alpha_multiplier": config.alpha_multiplier,
    #     "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
    #     "backup_entropy": config.backup_entropy,
    #     "policy_lr": config.policy_lr,
    #     "qf_lr": config.qf_lr,
    #     "bc_steps": config.bc_steps,
    #     "target_update_period": config.target_update_period,
    #     "cql_n_actions": config.cql_n_actions,
    #     "cql_importance_sample": config.cql_importance_sample,
    #     "cql_lagrange": config.cql_lagrange,
    #     "cql_target_action_gap": config.cql_target_action_gap,
    #     "cql_temp": config.cql_temp,
    #     "cql_alpha": config.cql_alpha,
    #     "cql_max_target_backup": config.cql_max_target_backup,
    #     "cql_clip_diff_min": config.cql_clip_diff_min,
    #     "cql_clip_diff_max": config.cql_clip_diff_max,
    #     # # New parameters for segment-based critic update
    #     # "use_segment_critic_update": config.use_segment_critic_update,
    #     # "epochs_critic": config.epochs_critic,
    #     # "segment_length": config.segment_length,
    #     # "num_segments": config.num_segments,
    #     # "n_step_return": config.n_step_return,
    #     # "clip_grad_norm": config.clip_grad_norm,
    #     # "use_mix_precision": config.use_mix_precision,
    #     # # New parameters for segment-based n-step return Q-learning
    #     # "use_segment_n_step_return_qf": config.use_segment_n_step_return_qf,
    #     # "return_type": config.return_type,
    # }

    print("---------------------------------------")
    print(f"Training CQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ContinuousCQL(**cql_kwargs) # modified for cql test

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    for t in range(int(config.max_timesteps)):
        # print("time_step", t)
        # batch = replay_buffer.sample(config.batch_size)
        # batch = replay_buffer_modular.sample(config.batch_size, normalize=True)
        # batch = convert_batch_dict_to_list(batch)
        # # print(batch[0].size())
        # batch = [b.to(config.device) for b in batch]

        batch_seq = replay_buffer_modular_seq.sample(config.batch_size, normalize=False)
        batch_seq = convert_batch_dict_to_list(batch_seq)
        batch_seq = [b.to(config.device) for b in batch_seq]

        log_dict = trainer.train(batch_seq)
        trainer.progress_bar.update(1)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                cql_actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            print("min score:", env.ref_min_score)
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")
            if config.checkpoints_path:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            wandb.log(
                {"d4rl_normalized_score": normalized_eval_score},
                step=trainer.total_it,
            )


def convert_batch_dict_to_list(batch: dict) -> List[torch.Tensor]:
    return [
        batch["observations"],
        batch["actions"],
        batch["rewards"],
        batch["next_observations"],
        batch["terminals"].float()  # make sure it's float if it's bool
    ]


if __name__ == "__main__":
    train()

