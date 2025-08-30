# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-71e5c35b-9eb6-a2a6-9b30-2e108de6212d,GPU-0073f373-55c9-4d6d-7621-ff5615e5f42d"
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

import algorithms.cw_offline.rl.replay_buffer as rb
import algorithms.cw_offline.rl.policy as pl
import algorithms.cw_offline.util as util
import algorithms.cw_offline.rl.critic.seq_critic as seq_critic

from torch.cuda.amp import GradScaler

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # wandb project name
    project: str = "TORL"
    # wandb group name
    group: str = "IQL-D4RL"
    # wandb run name
    name: str = "IQL_rp_a"
    # training dataset and evaluation environment
    env: str = "halfcheetah-medium-expert-v2"
    # discount factor
    discount: float = 0.99  # 0.99
    # coefficient for the target critic Polyak's update
    tau: float = 0.005
    # actor update inverse temperature, similar to AWAC
    # small beta -> BC, big beta -> maximizing Q-value
    beta: float = 3.0
    # coefficient for asymmetric critic loss
    # small tau -> optimistic, big beta -> conservative
    # (tau-1)
    iql_tau: float = 0.7 # 0.7
    # whether to use deterministic actor
    iql_deterministic: bool = False
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    # maximum size of the replay buffer
    buffer_size: int = 2_000_000
    # training batch size
    batch_size: int = 256
    # whether to normalize states
    normalize: bool = True
    # whether to normalize reward (like in IQL)
    normalize_reward: bool = False
    # V-critic function learning rate
    vf_lr: float = 3e-4
    # Q-critic learning rate
    qf_lr: float = 3e-4
    # actor learning rate
    actor_lr: float = 3e-4
    #  where to use dropout for policy network, optional
    actor_dropout: Optional[float] = None
    # evaluation frequency, will evaluate every eval_freq training steps
    eval_freq: int = int(1e3) # 5000
    # number of episodes to run during evaluation
    n_episodes: int = 10
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
    # file name for loading a model, optional
    load_model: str = ""
    # training random seed
    seed: int = 0
    # training device
    device: str = "cuda"

    lr_critic: float = 3e-4

    policy_lr: float = 3e-4

    random_target: bool = False

    clip_grad_norm: float = 0.0

    num_segments: int = 1

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
        entity="tyc1333",
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


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


def return_reward_range(dataset, max_episode_steps):
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


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        # # standard iql
        # q_network: nn.Module,
        # q_optimizer: torch.optim.Optimizer,
        # v_network: nn.Module,
        # v_optimizer: torch.optim.Optimizer,
        # TIQL
        critic: nn.Module,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
        # TIQL
        dtype: torch.dtype = torch.float32,
        wd_critic: int = 1e-5,
        lr_critic: int = 5e-5,
        betas: tuple = (0.9, 0.999),
        use_mix_precision: bool = False,
        soft_target_update_rate: float = 5e-3,
        target_update_period: int = 1,
        random_target: bool = True,
        clip_grad_norm : float = 1.0,
        num_segments: int = 1,
    ):
        self.max_action = max_action
        self.actor = actor
        # # standard iql
        # self.qf = q_network
        # self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        # self.vf = v_network
        # self.v_optimizer = v_optimizer
        # self.q_optimizer = q_optimizer
        # TIQL
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

        # added for TIQL
        self.dtype = dtype
        self.num_segments = num_segments
        # Initialize traj_length (will be set during training)
        self.traj_length = None


        self.wd_critic = wd_critic
        self.lr_critic = lr_critic
        self.betas = betas
        self.critic_1_optimizer, self.critic_2_optimizer = self.critic.configure_optimizer(
            weight_decay=self.wd_critic, learning_rate=self.lr_critic,
            betas=self.betas)
        self.critic_optimizer = (self.critic_1_optimizer, self.critic_2_optimizer)

        if not self.critic.single_q:
            self.critic_grad_scaler = [GradScaler(), GradScaler()]
        else:
            self.critic_grad_scaler = [GradScaler()] * 2

        self.discount_factor = torch.tensor(self.discount,
                                            dtype=self.dtype,
                                            device=self.device)
        self.use_mix_precision = use_mix_precision
        self.soft_target_update_rate = soft_target_update_rate
        self.target_update_period = target_update_period
        self.random_target = random_target

        self.log_now = False
        self.clip_grad_norm = clip_grad_norm


    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["net1_iql_loss"] = v_loss.item()
        log_dict["net1_adv"] = adv.mean().item()
        log_dict["net1_v"] = v.mean().item()
        log_dict[f"net1_v_target(q)_mean"] = target_q.mean().item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        # targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        targets = rewards + self.discount * next_v.detach()

        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        for i, q in enumerate(qs, start=1):
            log_dict[f"net{i}_q_loss"] = q_loss.item()
            log_dict[f"net{i}_q_mean"] = q.mean().item()
            log_dict[f"net{i}_q_target(r+v)_mean"] = targets.mean().item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        # policy_out = self.actor(observations)
        _, p_mean, p_std = self.actor.policy(observations)
        policy_out = Normal(p_mean, p_std)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

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
            # start_idx = 0

            start_idx = torch.randint(0, seg_length, [],
                                      dtype=torch.long, device=self.device)

        num_seg_actual = (self.traj_length - start_idx) // seg_length
        if pad_additional:
            num_seg_actual += 1

        idx_in_segments = torch.arange(0, num_seg_actual * seg_length,
                                       device=self.device) + start_idx
        idx_in_segments = idx_in_segments.view(-1, seg_length)
        idx_in_segments = torch.cat([idx_in_segments,
                                     idx_in_segments[:, -1:] + 1], dim=-1)

        if pad_additional and idx_in_segments[-1][0] == self.traj_length:
            return idx_in_segments[:-1]
        else:
            return idx_in_segments

    def segments_n_step_return_implicit_vf(self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            next_observations: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            # truncated: torch.Tensor,
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

        ##################### Compute the current n_step Q using dataset actions  ######################
        # [num_traj, num_segments, 1 + num_seg_actions]
        future_returns = torch.zeros([num_traj, num_segments,
                                      1 + num_seg_actions], device=self.device)

        # [num_traj, num_segments, dim_state]
        c_state = states[:, seg_start_idx]
        c_idx = util.add_expand_dim(seg_start_idx, [0], [num_traj])

        # actions in dataset
        # [num_traj, num_segments, segment_length, dim_actions]
        a_idx = idx_in_segments[:, :-1]
        action_pad_zero_end = torch.nn.functional.pad(actions, (0, 0, 0, num_seg_actions))
        d_actions = action_pad_zero_end[:, a_idx, :]
        a_idx = util.add_expand_dim(a_idx, [0], [num_traj])

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

        # try with current Q instead of target Q
        # with util.autocast_if(self.use_mix_precision):
        #     with torch.no_grad():
        #         cur_q1 = self.critic.critic(self.critic.net1,
        #                                     d_state=None, c_state=c_state,
        #                                     actions=d_actions, idx_d=None,
        #                                     idx_c=c_idx, idx_a=a_idx)
        #         if not self.critic.single_q:
        #             cur_q2 = self.critic.critic(self.critic.net2,
        #                                         d_state=None, c_state=c_state,
        #                                         actions=d_actions, idx_d=None,
        #                                         idx_c=c_idx, idx_a=a_idx)
        #         else:
        #             cur_q2 = cur_q1
        #
        #         # if self.random_target and not self.targets_overestimate:
        #         if self.random_target:
        #             # Randomly choose the Q value from Q1 or Q2
        #             mask = torch.randint(0, 2, cur_q1.shape, device=self.device)
        #             cur_q = cur_q1 * mask + cur_q2 * (1 - mask)
        #         else:
        #             cur_q = torch.minimum(cur_q1, cur_q2)
        #
        # # Use last q as the target of the V-func
        # # [num_traj, num_segments, 1 + num_seg_actions]
        # # -> [num_traj, num_segments]
        # # Tackle the Q-func beyond the length of the trajectory
        # # Option, Use the last q predictions
        # future_returns[:, :-1, 0] = cur_q[:, :-1, -1]
        #
        # # Find the idx where the action is the last action in the valid trajectory
        # last_valid_q_idx = idx_in_segments[-1] == traj_length
        # future_returns[:, -1, 0] = (
        #     cur_q[:, -1, last_valid_q_idx].squeeze(-1))

        ##################### Compute the V in the future ######################
        # state after executing the action
        # [num_traj, traj_length]
        c_idx = torch.arange(traj_length, device=self.device).long()
        c_idx = util.add_expand_dim(c_idx, [0], [num_traj])

        # [num_traj, traj_length, dim_state]
        n_state = next_observations  # [num_traj, traj_length, dim_state]

        # Use mix precision for faster computation
        with util.autocast_if(self.use_mix_precision):
            # [num_traj, traj_length]
            future_v1 = self.critic.critic(self.critic.net1,
                                           d_state=None, c_state=n_state,
                                           actions=None, idx_d=None,
                                           idx_c=c_idx, idx_a=None).squeeze(-1)
            if not self.critic.single_q:
                future_v2 = self.critic.critic(self.critic.net2,
                                               d_state=None, c_state=n_state,
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

        future_v = future_v.detach()

        # Pad zeros to the states which go beyond the traj length
        future_v_pad_zero_end \
            = torch.nn.functional.pad(future_v, (0, num_seg_actions))

        # [num_segments, num_seg_actions]
        v_idx = idx_in_segments[:, :-1]
        # assert v_idx.max() <= traj_length

        # [num_traj, traj_length] -> [num_traj, num_segments, num_seg_actions]
        future_returns[..., 1:] = future_v_pad_zero_end[:, v_idx]
        # for test using Q all equal 100
        # future_returns[future_returns != 0] = 100

        # if terminated, V should not be added to r+gamma*V, and truncated is already realized by set following seq as 0
        dones_pad_zero_end \
            = torch.nn.functional.pad(dones.squeeze(-1), (0, num_seg_actions))
        d_seq = dones_pad_zero_end[:, idx_in_segments[:, :-1]]
        future_returns[..., 1:] = (1-d_seq)*future_returns[..., 1:]

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
        if num_seg_actions > 1:
            discount_seq[-num_seg_actions + 1:] = 0

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

    # def train(self, batch: TensorBatch) -> Dict[str, float]:
    #     self.total_it += 1
    #     (
    #         observations_seq,
    #         actions_seq,
    #         rewards_seq,
    #         next_observations_seq,
    #         dones_seq,
    #     ) = batch
    #
    #     step_batch = [item[:, 0] for item in batch]
    #     (
    #         observations,
    #         actions,
    #         rewards,
    #         next_observations,
    #         dones,
    #     ) = step_batch
    #
    #     log_dict = {}
    #
    #     with torch.no_grad():
    #         next_v = self.vf(next_observations)
    #     # Update value function
    #     adv = self._update_v(observations, actions, log_dict)
    #
    #     rewards = rewards.squeeze(dim=-1)
    #     dones = dones.squeeze(dim=-1)
    #     # Update Q function
    #     self._update_q(next_v, observations, actions, rewards, dones, log_dict)
    #     # Update actor
    #     self._update_policy(adv, observations, actions, log_dict)
    #
    #     return log_dict

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations_seq,
            actions_seq,
            rewards_seq,
            next_observations_seq,
            dones_seq,
        ) = batch

        rewards_seq = rewards_seq.squeeze(-1)

        step_batch = [item[:, 0] for item in batch]
        (
            observations_step,
            actions_step,
            rewards_step,
            next_observations_step,
            dones_step,
        ) = step_batch

        log_dict = {}

        mse = torch.nn.MSELoss()
        # util.run_time_test(lock=True, key="update critic")
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

        # targets = None
        # vq_pred = None

        # Generate segments using the reusable get_segments method
        num_traj = observations_seq.shape[0]
        traj_length = observations_seq.shape[1]

        # Set traj_length for get_segments method
        self.traj_length = traj_length

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
                q_diff = vq_pred[..., 1:] - targets[..., 1:]
                q_sum = torch.sum(q_diff ** 2)
                # v_diff = vq_pred[..., 0] - targets[..., 0]
                adv = targets[..., 0] - vq_pred[..., 0]
                v_sum = torch.sum(adv ** 2)
                critic_loss = mse(vq_pred[..., 1:], targets[..., 1:])
                # print(vq_pred[..., 1:], targets[..., 1:], critic_loss, ((vq_pred[..., 1:]-targets[..., 1:])**2).mean())
                # asymmetric_l2_loss is different from the lecture
                # TODO: test and rewrite it as in the lecture
                iql_loss = asymmetric_l2_loss(adv, self.iql_tau)
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
                        f"{net_name}_q_target(r+v)_mean": targets[..., 1:].mean().item(),
                        f"{net_name}_return_mean": mc_returns_mean,
                        f"{net_name}_v": vq_pred[..., 0].mean().item(),
                        f"{net_name}_v_target(q)_mean": targets[..., 0].mean().item(),
                        f"{net_name}_adv": adv.mean().item(),
                        f"{net_name}_q_loss": critic_loss.item(),
                        f"{net_name}_iql_loss": iql_loss.item(),
                    }
                )

            # Update critic net parameters
            # util.run_time_test(lock=True, key="update critic net")
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

            # update_critic_net_time.append(
            #     util.run_time_test(lock=False,
            #                        key="update critic net"))

            # Logging
            critic_loss_list.append(critic_loss.item())
            # critic_grad_norm.append(grad_norm)
            # clipped_critic_grad_norm.append(grad_norm_c)

            # Update target network
            # util.run_time_test(lock=True, key="copy critic net")
            self.critic.update_target_net(net, target_net)
            # update_target_net_time.append(
            #     util.run_time_test(lock=False,
            #                        key="copy critic net"))

        # IQL policy update
        self.critic.eval()  # disable dropout
        self.critic.requires_grad(False)
        with torch.no_grad():
            # online V predictions from both critics (average them to update policy)
            v1_online = self.critic.critic(self.critic.net1,
                                           d_state=None, c_state=c_state,
                                           actions=seg_actions, idx_d=None, idx_c=seg_start_idx, idx_a=seg_actions_idx)[..., 0]
            if self.critic.single_q:
                v_avg = v1_online
            else:
                v2_online = self.critic.critic(self.critic.net2,
                                               d_state=None, c_state=c_state,
                                               actions=seg_actions, idx_d=None, idx_c=seg_start_idx, idx_a=seg_actions_idx)[..., 0]
                # v_avg = 0.5 * (v1_online + v2_online)
                v_avg = torch.minimum(v1_online, v2_online)
            q1_target = self.critic.critic(self.critic.target_net1,
                                           d_state=None, c_state=c_state,
                                           actions=seg_actions, idx_d=None, idx_c=seg_start_idx,
                                           idx_a=seg_actions_idx)[..., 1]
            if self.critic.single_q:
                q_target_avg = q1_target
            else:
                q2_target = self.critic.critic(self.critic.target_net2,
                                               d_state=None, c_state=c_state,
                                               actions=seg_actions, idx_d=None, idx_c=seg_start_idx,
                                               idx_a=seg_actions_idx)[..., 1]
                # q_target_avg = 0.5 * (q1_target + q2_target)
                q_target_avg = torch.minimum(q1_target, q2_target)

        log_dict.update(
            {
                "pure_v_pred_avg": v_avg.mean().item(),
            }
        )

        # TODO: check the policy update
        # adv = (targets[..., 1] - v_avg).detach()
        adv = (q_target_avg - v_avg).detach()
        adv_np = adv.detach().cpu().numpy()
        adv_p95 = np.percentile(adv_np, 95)
        adv_std = adv_np.std()
        adv_min = adv_np.min()
        adv_max = adv_np.max()
        log_dict.update({"1step_adv": adv.mean().item(),
                         "adv_p95": float(adv_p95),
                         "adv_std": float(adv_std),
                         "adv_min": float(adv_min),
                         "adv_max": float(adv_max),
                         })  # should >0
        action_start_idx = idx_in_segments[..., 0]
        c_action = actions_seq[:, action_start_idx]
        self._update_policy(adv, c_state, c_action, log_dict)

        # if self.total_it % self.target_update_period == 0:
        #     self.update_target_network(self.soft_target_update_rate)

        return log_dict

        # with torch.no_grad():
        #     next_v = self.vf(next_observations)
        # # Update value function
        # adv = self._update_v(observations, actions, log_dict)
        # rewards = rewards.squeeze(dim=-1)
        # dones = dones.squeeze(dim=-1)
        # # Update Q function
        # self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # # Update actor
        # self._update_policy(adv, observations, actions, log_dict)
        #
        # return log_dict

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.critic.target_net1, self.critic.net1, soft_target_update_rate)
        soft_update(self.critic.target_net2, self.critic.net2, soft_target_update_rate)

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

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)

    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    # replay_buffer = ReplayBuffer(
    #     state_dim,
    #     action_dim,
    #     config.buffer_size,
    #     config.device,
    # )
    # replay_buffer.load_d4rl_dataset(dataset)

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

    replay_buffer_modular_seq = rb.SeqReplayBuffer(
        replay_buffer_data_shape,
        replay_buffer_norm_info,
        config.buffer_size,
        device=config.device,
        env_name = config.env,
    )

    replay_buffer_modular_seq.load_d4rl_dataset(dataset)
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

    # q_network = TwinQ(state_dim, action_dim).to(config.device)
    # v_network = ValueFunction(state_dim).to(config.device)
    # actor = (
    #     DeterministicPolicy(
    #         state_dim, action_dim, max_action, dropout=config.actor_dropout
    #     )
    #     if config.iql_deterministic
    #     else GaussianPolicy(
    #         state_dim, action_dim, max_action, dropout=config.actor_dropout
    #     )
    # ).to(config.device)
    # v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    # q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    # actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    critic_config = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "single_q": False,
        "device": config.device,
        "dtype": "float32",
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

    critic = seq_critic.SeqCritic(**critic_config)

    policy_kwargs = {
        "mean_net_args": {
            "avg_neuron": 256,
            "num_hidden": 2,
            "shape": 0.0,
        },
        "variance_net_args": {
            "std_only": True,
            "contextual": False,  # state-independent parameter
            # "avg_neuron": 256,
            # "num_hidden": 3,
            # "shape": 0.0,
        },
        "init_method": "orthogonal",  # "orthogonal",
        "out_layer_gain": 1.0,  # 0.01,
        # "min_std": 1e-5,
        "act_func_hidden": "relu",
        "act_func_last": "tanh",
    }

    actor = pl.GaussianPolicy(
        state_dim,
        action_dim,
        max_action=max_action,
        # log_std_multiplier=1.0,
        # orthogonal_init=False,
        device=config.device,
        **policy_kwargs,
    )

    actor_optimizer = torch.optim.Adam(actor.parameters, config.policy_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        # # standard IQL
        # "q_network": q_network,
        # "q_optimizer": q_optimizer,
        # "v_network": v_network,
        # "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "critic": critic,
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
        "lr_critic": config.lr_critic,
        # TIQL
        "random_target": config.random_target,
        "clip_grad_norm": config.clip_grad_norm,
        "num_segments": config.num_segments,
    }

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    for t in range(int(config.max_timesteps)):
        # batch = replay_buffer.sample(config.batch_size)
        # batch = [b.to(config.device) for b in batch]

        batch_seq = replay_buffer_modular_seq.sample(config.batch_size, normalize=False)
        # if batch_seq["truncated"].any().item():
        #     print("batch contains truncated!")
        batch_seq = convert_batch_dict_to_list(batch_seq)
        batch_seq = [b.to(config.device) for b in batch_seq]

        log_dict = trainer.train(batch_seq)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                actor,
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
            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            wandb.log(
                {"d4rl_normalized_score": normalized_eval_score}, step=trainer.total_it
            )

def convert_batch_dict_to_list(batch: dict) -> List[torch.Tensor]:
    return [
        batch["observations"],
        batch["actions"],
        batch["rewards"],
        batch["next_observations"],
        batch["terminals"].float(),  # make sure it's float if it's bool
    ]



if __name__ == "__main__":
    train()
