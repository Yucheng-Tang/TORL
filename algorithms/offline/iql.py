# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
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
    discount: float = 0.99
    # coefficient for the target critic Polyak's update
    tau: float = 0.005
    # actor update inverse temperature, similar to AWAC
    # small beta -> BC, big beta -> maximizing Q-value
    beta: float = 3.0
    # coefficient for asymmetric critic loss
    iql_tau: float = 0.7
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
    eval_freq: int = int(5e3) # 5000
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

    policy_lr: float = 3e-4

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
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
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
        log_dict["q_loss"] = q_loss.item()
        for i, q in enumerate(qs, start=1):
            log_dict[f"q{i}"] = q.mean().item()
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
                                      dtype=torch.long, device=self._device)

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
        action_pad_zero_end = torch.nn.functional.pad(actions, (0, 0, 0, num_seg_actions))
        d_actions = action_pad_zero_end[:, a_idx, :]
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

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations_seq,
            actions_seq,
            rewards_seq,
            next_observations_seq,
            dones_seq,
        ) = batch

        step_batch = [item[:, 0] for item in batch]
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = step_batch

        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

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

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    # actor = (
    #     DeterministicPolicy(
    #         state_dim, action_dim, max_action, dropout=config.actor_dropout
    #     )
    #     if config.iql_deterministic
    #     else GaussianPolicy(
    #         state_dim, action_dim, max_action, dropout=config.actor_dropout
    #     )
    # ).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    # actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

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
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
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
        batch["terminals"].float()  # make sure it's float if it's bool
    ]



if __name__ == "__main__":
    train()
