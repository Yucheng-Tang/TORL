import copy
import math

import numpy as np
import torch

import algorithms.cw_offline.rl.critic.abstract_critic as abs_critic
import algorithms.cw_offline.rl.policy.abstract_policy as abs_policy
import algorithms.cw_offline.rl.sampler.abstract_sampler as abs_sampler
import algorithms.cw_offline.util as util
from algorithms.cw_offline.rl.agent import AbstractAgent
from algorithms.cw_offline.rl.critic import SeqCritic
from algorithms.cw_offline.rl.replay_buffer import SeqReplayBuffer
from algorithms.cw_offline.util import autocast_if

from trust_region_projections.utils.projection_utils import gaussian_kl_details

from torch.cuda.amp import autocast, GradScaler


class SeqQAgent(AbstractAgent):
    def __init__(self,
                 policy: abs_policy.AbstractGaussianPolicy,
                 critic: SeqCritic,
                 sampler: abs_sampler.AbstractSampler,
                 replay_buffer: SeqReplayBuffer,
                 projection=None,
                 dtype=torch.float32,
                 device=torch.device("cpu"),
                 **kwargs):

        self.segments_config = kwargs.get("segments_config")
        self.traj_length = kwargs.get("traj_length")

        self.betas = kwargs.get("betas", (0.9, 0.999))

        super().__init__(policy, critic, sampler, projection,
                         dtype=dtype, device=device, **kwargs)

        self.clip_critic = float(kwargs.get("clip_critic", 0.0))
        self.clip_grad_norm = float(kwargs.get("clip_grad_norm", 0.0))
        self.norm_q_loss = kwargs.get("norm_q_loss", False)
        self.clip_advantages = kwargs.get("clip_advantages", False)
        self.entropy_penalty_coef = float(kwargs.get("entropy_penalty_coef",
                                                     0.0))
        self.set_variance = kwargs.get("set_variance", False)
        self.balance_check = kwargs.get("balance_check", 10)
        self.evaluation_interval = kwargs.get("evaluation_interval", 1)
        self.log_now = False
        # For off-policy learning
        self.replay_buffer = replay_buffer
        self.return_type = kwargs.get("return_type")
        self.batch_size = kwargs.get("batch_size")

        self.critic_update_from = kwargs.get("critic_update_from", 0)
        self.policy_update_from = kwargs.get("policy_update_from", 0)
        assert self.critic_update_from <= self.policy_update_from, \
            "Critic update should be earlier than policy update"
        self.random_target = kwargs.get("random_target", True)
        self.type_policy_q_update = kwargs.get("type_policy_q_update", "mean")
        self.num_samples_in_targets = kwargs.get("num_samples_in_targets", 1)
        self.use_old_policy = kwargs.get("use_old_policy", False)
        if self.use_old_policy:
            self.old_policy_update_rate = kwargs.get("old_policy_update_rate", 1)
            self.policy_old = copy.deepcopy(self.policy)
        self.fresh_agent = True
        self.targets_overestimate = False
        self.traj_has_downsample = (
                self.sampler.traj_downsample_factor is not None)
        self.norm_data = self.replay_buffer.has_normalizer

        # Float 16
        self.use_mix_precision = kwargs.get("use_mix_precision", False)
        self.policy_grad_scaler = GradScaler()
        if not self.critic.single_q:
            self.critic_grad_scaler = [GradScaler(), GradScaler()]
        else:
            self.critic_grad_scaler = [GradScaler()] * 2

    def get_optimizer(self, policy, critic):
        """
        Get the policy and critic network optimizers

        Args:
            policy: policy network
            critic: critic network

        Returns:
            two optimizers
        """
        self.policy_net_params = policy.parameters
        policy_optimizer = torch.optim.Adam(params=self.policy_net_params,
                                            lr=self.lr_policy,
                                            weight_decay=self.wd_policy)
        critic_opt1, critic_opt2 = self.critic.configure_optimizer(
            weight_decay=self.wd_critic, learning_rate=self.lr_critic,
            betas=self.betas)

        return policy_optimizer, (critic_opt1, critic_opt2)

    def step(self):
        # Update total step count
        self.num_iterations += 1

        # If logging data in the current step
        self.log_now = self.evaluation_interval == 1 or \
                       self.num_iterations % self.evaluation_interval == 1
        update_critic_now = self.num_iterations >= self.critic_update_from
        update_policy_now = self.num_iterations >= self.policy_update_from

        if not self.fresh_agent:
            buffer_is_ready = self.replay_buffer.is_full()
            self.num_iterations -= 1  # iteration only for collecting data
        else:
            buffer_is_ready = True
        # Collect dataset
        util.run_time_test(lock=True, key="sampling")
        dataset, num_env_interation = \
            self.sampler.run(training=True, policy=self.policy,
                             critic=self.critic)
        self.num_global_steps += num_env_interation
        sampling_time = util.run_time_test(lock=False, key="sampling")

        # Process dataset and save to RB
        util.run_time_test(lock=True, key="process_dataset")
        dataset = self.process_dataset(dataset)
        process_dataset_time = util.run_time_test(lock=False,
                                                  key="process_dataset")

        # Update agent, if continue training, collect dataset first
        if update_critic_now and buffer_is_ready:
            # Update agent
            util.run_time_test(lock=True, key="update")

            critic_loss_dict, policy_loss_dict = self.update(update_policy_now)

            if update_critic_now and self.schedule_lr_critic:
                lr_schedulers = util.make_iterable(self.critic_lr_scheduler)
                for scheduler in lr_schedulers:
                    scheduler.step()

            if update_policy_now and self.schedule_lr_policy:
                self.policy_lr_scheduler.step()

            update_time = util.run_time_test(lock=False, key="update")

        else:
            critic_loss_dict, policy_loss_dict = {}, {}
            update_time = 0

        # Log data
        if self.log_now and buffer_is_ready:
            # Generate statistics for environment rollouts
            dataset_stats = \
                util.generate_many_stats(dataset, "exploration", to_np=True,
                                         exception_keys=["decision_idx"])

            # Prepare result metrics
            result_metrics = {
                **dataset_stats,
                "sampling_time": sampling_time,
                "process_dataset_time": process_dataset_time,
                "num_global_steps": self.num_global_steps,
                **critic_loss_dict, **policy_loss_dict,
                "update_time": update_time,
                "lr_policy": self.policy_lr_scheduler.get_last_lr()[0]
                if self.schedule_lr_policy else self.lr_policy,
                "lr_critic1": self.critic_lr_scheduler[0].get_last_lr()[0]
                if self.schedule_lr_critic else self.lr_critic,
                "lr_critic2": self.critic_lr_scheduler[1].get_last_lr()[0]
                if self.schedule_lr_critic else self.lr_critic
            }

            # Evaluate agent
            util.run_time_test(lock=True)
            evaluate_metrics = util.generate_many_stats(
                self.evaluate()[0], "evaluation", to_np=True,
                exception_keys=["decision_idx"])
            evaluation_time = util.run_time_test(lock=False)
            result_metrics.update(evaluate_metrics),
            result_metrics.update({"evaluation_time": evaluation_time}),
        else:
            result_metrics = {}

        return result_metrics

    def process_dataset(self, dataset):
        self.replay_buffer.add(dataset)
        return dataset

    def get_segments(self, pad_additional=False):
        # New implementation
        num_seg = self.segments_config["num_segments"]
        if isinstance(num_seg, int):
            pass
        elif isinstance(num_seg, str):
            if num_seg == "random":
                possible_num_segments = torch.arange(1, 26, device=self.device)
                segment_lengths = self.traj_length // possible_num_segments
                segment_lengths_unique = segment_lengths.unique()
                possible_num_segments_after_unique\
                    = self.traj_length // segment_lengths_unique
                # random choose the number of segments
                num_seg = possible_num_segments_after_unique[torch.randint(
                    0, len(possible_num_segments_after_unique), [])]

                # num_seg = torch.randint(1, 26, [],
                #                         dtype=torch.long, device=self.device)
            else:
                raise ValueError("Invalid num_seg")
        seg_length = self.traj_length // num_seg
        if num_seg == 1:
            start_idx = 0
        else:
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

    def make_new_pred(self, dataset, **kwargs):
        if self.projection is None:
            states = dataset["step_states"]
            decision_idx = dataset["decision_idx"]
            num_traj = states.shape[0]
            d_state = states[
                torch.arange(num_traj, device=self.device), decision_idx]
            params_mean_new, params_L_new = self.policy.policy(d_state)
            info = {}
            return params_mean_new, params_L_new, info
        else:
            tr_kl_dict = kwargs.get("tr_kl_dict", None)
            compute_trust_region_loss = kwargs.get("compute_trust_region_loss",
                                                   True)
            states = dataset["step_states"]

            # [num_traj]
            decision_idx = dataset["decision_idx"]

            num_traj = states.shape[0]

            # Decision state
            # [num_traj, dim_state]
            d_state = states[
                torch.arange(num_traj, device=self.device), decision_idx]
            if self.use_old_policy:
                with torch.no_grad():
                    params_mean_old, params_L_old = self.policy_old.policy(d_state)
            else:
                params_mean_old = dataset["segment_params_mean"]
                params_L_old = dataset["segment_params_L"]

            # entropy decay
            if self.projection.initial_entropy is None:
                self.projection.initial_entropy = \
                    self.policy.entropy([params_mean_old, params_L_old]).mean()

            # Make prediction
            params_mean_new, params_L_new = self.policy.policy(d_state)

            # Projection
            util.run_time_test(lock=True, key="projection")
            proj_mean, proj_L = self.projection(self.policy,
                                                (params_mean_new, params_L_new),
                                                (params_mean_old, params_L_old),
                                                self.num_iterations)
            projection_time = util.run_time_test(lock=False, key="projection")

            # Log KL of: old-new, old-proj, new-proj
            if self.log_now and tr_kl_dict is not None:
                tr_kl_dict = self.kl_old_new_proj(params_mean_new, params_L_new,
                                                  params_mean_old, params_L_old,
                                                  proj_mean, proj_L, tr_kl_dict)
            else:
                tr_kl_dict = {}

            # Trust Region loss
            if compute_trust_region_loss:
                trust_region_loss = \
                    self.projection.get_trust_region_loss(self.policy,
                                                          (params_mean_new,
                                                           params_L_new),
                                                          (proj_mean, proj_L),
                                                          set_variance=
                                                          self.set_variance)
            else:
                trust_region_loss = None

            info = {"trust_region_loss": trust_region_loss,
                    "projection_time": projection_time,
                    "tr_kl_dict": tr_kl_dict}

            return proj_mean, proj_L, info

    def segments_n_step_return_vf(self, dataset, idx_in_segments):
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
        states = dataset["step_states"]  # [num_traj, traj_length, dim_state]
        rewards = dataset["step_rewards"]
        dones = dataset["step_dones"]
        decision_idx = dataset["decision_idx"]  # [num_traj]
        # [num_traj, traj_length, dim_action]
        traj_init_pos = dataset["step_desired_pos"]
        traj_init_vel = dataset["step_desired_vel"]

        num_segments = idx_in_segments.shape[0]
        num_seg_actions = idx_in_segments.shape[-1] - 1
        seg_start_idx = idx_in_segments[..., 0]

        num_traj, traj_length = states.shape[0], states.shape[1]

        # NOTE: additional dimension is defined as [num_traj, num_segments]

        with torch.no_grad():
            # params: [num_traj, num_weights]
            params_mean_new, params_L_new, _ \
                = self.make_new_pred(dataset, compute_trust_region_loss=False)

        # [num_traj, num_weights] -> [num_traj, num_segments, num_weights]
        params_mean_new = util.add_expand_dim(params_mean_new, [1],
                                              [num_segments])
        params_L_new = util.add_expand_dim(params_L_new, [1],
                                           [num_segments])


        # [num_traj, traj_length]
        times = self.sampler.get_times(dataset["segment_init_time"],
                                       self.sampler.num_times,
                                       self.traj_has_downsample)
        # [num_traj, traj_length] -> [num_traj, num_segments]
        times = times[:, seg_start_idx]

        # [num_traj, num_segments] -> [num_traj, num_segments, num_seg_actions]
        action_times = util.add_expand_dim(times, [-1], [num_seg_actions])
        time_evo = torch.linspace(0, self.sampler.dt * (num_seg_actions-1),
                                  num_seg_actions, device=self.device).float()
        action_times = action_times + time_evo

        # Init time, shape: [num_traj, num_segments]
        init_time = times - self.sampler.dt

        # [num_traj, num_segments, dim_action]
        init_pos = traj_init_pos[:, seg_start_idx]
        init_vel = traj_init_vel[:, seg_start_idx]

        # Get the new MP trajectory using the current policy and condition on
        # the buffered desired pos and vel as initial conditions
        # util.set_global_random_seed(0)

        params_mean_new = self.fix_relative_goal_for_segments(
            params=params_mean_new, traj_init_pos=traj_init_pos[:, 0],
            segments_init_pos=init_pos)

        # [num_traj, num_segments, num_seg_actions, dim_action] or
        # [num_traj, num_segments, num_smps, num_seg_actions, dim_action]
        actions = self.policy.sample(require_grad=False,
                                     params_mean=params_mean_new,
                                     params_L=params_L_new, times=action_times,
                                     init_time=init_time,
                                     init_pos=init_pos, init_vel=init_vel,
                                     use_mean=False,
                                     num_samples=self.num_samples_in_targets)

        # Normalize actions
        if self.norm_data:
            actions = self.replay_buffer.normalize_data("step_actions", actions)

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
        c_state = states[:, seg_start_idx]

        if self.num_samples_in_targets == 1:
            # [num_traj, num_segments]
            c_idx = util.add_expand_dim(seg_start_idx, [0], [num_traj])

            # [num_segments, num_seg_actions]
            # -> [num_traj, num_segments, num_seg_actions]
            a_idx = util.add_expand_dim(idx_in_segments[..., :-1],
                                        [0], [num_traj])
        else:
            num_smp = self.num_samples_in_targets

            # -> [num_traj, num_segments, num_smp, dim_state]
            c_state = util.add_expand_dim(c_state, [2], [num_smp])

            # [num_traj, num_segments, num_smp]
            c_idx = util.add_expand_dim(seg_start_idx, [0, 2],
                                        [num_traj, num_smp])
            # [num_segments, num_seg_actions]
            # -> [num_traj, num_segments, num_smp, num_seg_actions]
            a_idx = util.add_expand_dim(idx_in_segments[..., :-1],
                                        [0, 2],
                                        [num_traj, num_smp])

        # Use mix precision for faster computation
        with autocast_if(self.use_mix_precision):
            # [num_traj, num_segments, (num_smp,) 1 + num_seg_actions]
            future_q1 = self.critic.critic(self.critic.target_net1,
                                           d_state=None, c_state=c_state,
                                           actions=actions, idx_d=None,
                                           idx_c=c_idx, idx_a=a_idx)
            if not self.critic.single_q:
                future_q2 = self.critic.critic(self.critic.target_net2,
                                               d_state=None, c_state=c_state,
                                               actions=actions, idx_d=None,
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

        if self.num_samples_in_targets > 1:
            future_q = future_q.mean(dim=-2)

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

        # Option, Use the average q predictions, exclude the vf at 0th place
        # Option does not work well
        # future_returns[:, :-1, 0] = future_q[:, :-1, 1:].mean(dim=-1)

        # Find the idx where the action is the last action in the valid trajectory
        # last_valid_q_idx = (idx_in_segments[-1] <= traj_length)[1:]
        # future_returns[:, -1, 0] = future_q[:, -1, 1:][..., last_valid_q_idx].mean(dim=-1)


        ##################### Compute the V in the future ######################
        # state after executing the action
        # [num_traj, traj_length]
        c_idx = torch.arange(traj_length, device=self.device).long()
        c_idx = util.add_expand_dim(c_idx, [0], [num_traj])

        # [num_traj, traj_length, dim_state]
        c_state = states

        # Use mix precision for faster computation
        with autocast_if(self.use_mix_precision):
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

    def segments_n_step_return_qf(self, dataset, idx_in_segments):
        """
        Segment-wise n-step return computation

        Args:
            dataset:
            idx_in_segments:

        Returns:
            n_step_returns: [num_traj, num_segments, 1 + num_seg_actions]

        """
        states = dataset["step_states"]  # [num_traj, traj_length, dim_state]
        rewards = dataset["step_rewards"]
        dones = dataset["step_dones"]
        decision_idx = dataset["decision_idx"]  # [num_traj]
        # [num_traj, traj_length, dim_action]
        init_pos = dataset["step_desired_pos"]
        init_vel = dataset["step_desired_vel"]

        num_segments = idx_in_segments.shape[0]
        num_seg_actions = idx_in_segments.shape[-1] - 1
        seg_start_idx = idx_in_segments[..., 0]

        num_traj, traj_length = states.shape[0], states.shape[1]

        with torch.no_grad():
            # params: [num_traj, num_weights]
            params_mean_new, params_L_new, _ \
                = self.make_new_pred(dataset, compute_trust_region_loss=False)

        # NOTE: additional dimension is defined as [num_traj, traj_length]
        # [num_traj, traj_length]
        times = self.sampler.get_times(dataset["segment_init_time"],
                                       self.sampler.num_times,
                                       self.traj_has_downsample)

        # params: [num_traj, num_weights]
        # -> [num_traj, traj_length, num_weights]
        params_mean_new = util.add_expand_dim(params_mean_new, [1],
                                              [traj_length])
        params_L_new = util.add_expand_dim(params_L_new, [1],
                                           [traj_length])

        # Init time, shape: [num_traj, traj_length]
        init_time = times - self.sampler.dt

        # [num_traj, traj_length] -> [num_traj, traj_length, num_seg_actions]
        action_times = util.add_expand_dim(times, [-1], [num_seg_actions])
        time_evo = torch.linspace(0, self.sampler.dt * (num_seg_actions-1),
                                num_seg_actions, device=self.device).float()
        action_times = action_times + time_evo

        # Get the new MP trajectory using the current policy and condition on
        # the buffered desired pos and vel as initial conditions
        # [num_traj, traj_length, num_seg_actions, dim_action]
        # util.set_global_random_seed(0)

        params_mean_new = self.fix_relative_goal_for_segments(
            params=params_mean_new, traj_init_pos=init_pos[:, 0],
            segments_init_pos=init_pos)

        # [num_traj, traj_length, num_seg_actions, dim_action] or
        # [num_traj, traj_length, num_smps, num_seg_actions, dim_action]
        actions = self.policy.sample(require_grad=False,
                                     params_mean=params_mean_new,
                                     params_L=params_L_new, times=action_times,
                                     init_time=init_time,
                                     init_pos=init_pos, init_vel=init_vel,
                                     use_mean=False,
                                     num_samples=self.num_samples_in_targets)

        # Normalize actions
        if self.norm_data:
            actions = self.replay_buffer.normalize_data("step_actions", actions)

        ################### Compute the Q in the future ####################
        # num_traj, traj_length = actions.shape[0], actions.shape[1]

        future_q = torch.zeros([num_traj, traj_length],
                               device=self.device)

        # [num_traj, dim_state] -> [num_traj, traj_length, dim_state]
        # d_state = states[
        #     torch.arange(num_traj, device=self.device), decision_idx]
        # d_state = util.add_expand_dim(d_state, [1], [traj_length])

        # [num_traj] -> [num_traj, traj_length]
        # d_idx = util.add_expand_dim(decision_idx, [1], [traj_length])

        # [num_traj, traj_length, dim_state]
        c_state = states

        # [num_times]
        c_idx = torch.arange(traj_length, device=self.device).long()

        if self.num_samples_in_targets == 1:
            # [num_times] -> [num_traj, traj_length]
            c_idx = util.add_expand_dim(c_idx, [0], [num_traj])
        else:
            num_smp = self.num_samples_in_targets

            # [num_times] -> [num_traj, traj_length, num_smp]
            c_idx = util.add_expand_dim(c_idx, [0, 2], [num_traj, num_smp])

        # [num_traj, traj_length]
        # -> [num_traj, traj_length, num_seg_actions]
        # or if num_smp >1, then
        # [num_traj, traj_length, num_smp]
        # -> [num_traj, traj_length, num_smp, num_seg_actions]
        a_idx = util.add_expand_dim(c_idx, [-1], [num_seg_actions])
        a_idx_evo = torch.arange(num_seg_actions, device=self.device).long()
        a_idx = a_idx + a_idx_evo

        # Use mix precision for faster computation
        with autocast_if(self.use_mix_precision):
            #  v + [q, q, ..., q]
            # [num_traj, traj_length, (num_smp,) 1 + num_seg_actions],
            future_q1 = self.critic.critic(self.critic.target_net1,
                                           d_state=None, c_state=c_state,
                                           actions=actions, idx_d=None,
                                           idx_c=c_idx, idx_a=a_idx)
            if not self.critic.single_q:
                future_q2 = self.critic.critic(self.critic.target_net2,
                                               d_state=None, c_state=c_state,
                                               actions=actions, idx_d=None,
                                               idx_c=c_idx, idx_a=a_idx)
            else:
                future_q2 = future_q1

        # if self.random_target and not self.targets_overestimate:
        if self.random_target:
            # Randomly choose the Q value from Q1 or Q2
            mask = torch.randint(0, 2, future_q1.shape, device=self.device)
            min_q1_q2 = future_q1 * mask + future_q2 * (1 - mask)
        else:
            # [num_traj, traj_length, (num_smp,) 1 + num_seg_actions]
            min_q1_q2 = torch.minimum(future_q1, future_q2)

        if self.num_samples_in_targets > 1:
            min_q1_q2 = min_q1_q2.mean(dim=-2)

        # Note: Tackle the Q-func beyond the length of the trajectory
        ## Use the last q in the sequence if time is within the traj length
        future_q[:, :-num_seg_actions] \
            = min_q1_q2[:, :-num_seg_actions, -1]

        ## Use the q of the last traj time if actions is beyond the traj length
        for i in range(num_seg_actions):
            future_q[:, -num_seg_actions + i] \
                = min_q1_q2[:, -num_seg_actions + i, -1 - i]

        ################ Compute the reward in the future ##################
        # [num_traj, traj_length] -> [num_traj, traj_length + 1]
        future_r_pad_zero_end \
            = torch.nn.functional.pad(rewards, (0, num_seg_actions))

        # discount_seq as: [1, gamma, gamma^2..., 0]
        discount_idx \
            = torch.arange(traj_length + num_seg_actions, device=self.device)
        discount_seq = self.discount_factor.pow(discount_idx)
        discount_seq[-num_seg_actions:] = 0

        # Apply discount to all rewards and returns w.r.t the traj start
        # [num_traj, traj_length + 1]
        future_q_pad_zero_end = torch.nn.functional.pad(future_q,
                                                        (0, num_seg_actions))
        discount_q = future_q_pad_zero_end * discount_seq

        # [num_traj, traj_length + 1]
        discount_r = future_r_pad_zero_end * discount_seq

        # -> [num_traj, num_segments, 1 + traj_length]
        discount_r = util.add_expand_dim(discount_r, [1],
                                         [num_segments])

        # [num_traj, num_segments, num_seg_actions + 1]
        seg_discount_q = discount_q[:, idx_in_segments]

        # -> [num_traj, num_segments, num_seg_actions + 1]
        seg_reward_idx = util.add_expand_dim(idx_in_segments, [0],
                                             [num_traj])

        # torch.gather shapes
        # input: [num_traj, num_segments, traj_length + 1]
        # index: [num_traj, num_segments, num_seg_actions + 1]
        # result: [num_traj, num_segments, num_seg_actions + 1]
        seg_discount_r = torch.gather(input=discount_r, dim=-1,
                                      index=seg_reward_idx)

        # [num_traj, num_segments, num_seg_actions + 1] ->
        # [num_traj, num_segments, num_seg_actions + 1, num_seg_actions + 1]
        seg_discount_r = util.add_expand_dim(seg_discount_r, [-2],
                                             [num_seg_actions + 1])

        # Get a lower triangular mask with off-diagonal elements as 1
        # [num_seg_actions + 1, num_seg_actions + 1]
        reward_tril_mask = torch.tril(torch.ones(num_seg_actions + 1,
                                                 num_seg_actions + 1,
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

    def compute_true_return(self, dataset):
        """
        Compute the true return for the batch of trajectories

        Args:
            dataset:sampled trajectories

        Returns:
            true returns: [num_traj, traj_length]

        """
        rewards = dataset["step_rewards"]
        dones = dataset["step_dones"]

        values = torch.zeros([*(rewards.shape[:-1]), rewards.shape[-1] + 1],
                             device=self.device, dtype=self.dtype)

        time_limit_dones = torch.zeros_like(dones)

        # [num_traj, traj_length]
        _, returns = util.get_advantage_return(rewards, values, dones,
                                               time_limit_dones,
                                               self.discount_factor,
                                               False, None)

        return returns

    def update_critic(self):
        raise NotImplementedError

    def update_policy(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, update_policy=True):
        """
        update critic and policy

        Returns:
            critic_loss_dict: critic loss dictionary
            policy_loss_dict: policy loss dictionary
        """
        # define critic loss
        mse = torch.nn.MSELoss()

        # critic loss log
        critic_loss_list = []
        critic_grad_norm = []
        clipped_critic_grad_norm = []
        update_critic_net_time = []
        update_target_net_time = []

        # Value estimation error log
        mc_returns_list = []
        targets_list = []
        targets_bias_list = []

        ########################################################################
        #                             Update critic
        ########################################################################
        util.run_time_test(lock=True, key="update critic")

        self.critic.train()
        self.critic.requires_grad(True)

        for grad_idx in range(self.epochs_critic):
            # Sample from replay buffer
            dataset = self.replay_buffer.sample(
                self.batch_size, normalize=self.norm_data,
                use_priority=self.replay_buffer.has_priority,
                policy_recent=False)
            states = dataset["step_states"]
            actions = dataset["step_actions"]
            num_traj = states.shape[0]
            traj_length = states.shape[1]

            # Note: Use N-step Q-func return for segment-wise update
            if self.return_type == "segment_n_step_return_qf":
                idx_in_segments = self.get_segments(pad_additional=True)
                seg_start_idx = idx_in_segments[..., 0]
                assert seg_start_idx[-1] < self.traj_length
                seg_actions_idx = idx_in_segments[..., :-1]
                num_seg_actions = seg_actions_idx.shape[-1]

                # [num_traj, num_segments, dim_state]
                c_state = states[:, seg_start_idx]
                seg_start_idx = util.add_expand_dim(seg_start_idx, [0], [num_traj])

                padded_actions = torch.nn.functional.pad(
                    actions, (0, 0, 0, num_seg_actions), "constant", 0)

                # [num_traj, num_segments, num_seg_actions, dim_action]
                seg_actions = padded_actions[:, seg_actions_idx]

                # [num_traj, num_segments, num_seg_actions]
                seg_actions_idx = util.add_expand_dim(seg_actions_idx, [0],
                                                      [num_traj])

                # [num_traj, num_segments, 1 + num_seg_actions]
                targets = self.segments_n_step_return_qf(dataset, idx_in_segments)

                # Log targets and MC returns
                # if self.log_now:
                mc_returns_mean = util.compute_mc_return(
                    dataset["step_rewards"].mean(dim=0),
                    self.discount_factor).mean().item()

                mc_returns_list.append(mc_returns_mean)
                targets_mean = targets.mean().item()
                targets_list.append(targets_mean)
                targets_bias_list.append(targets_mean - mc_returns_mean)

                # for net, target_net, opt in self.critic_nets_and_opt():
                for net, target_net, opt, scaler in self.critic_nets_and_opt():

                    # Use mix precision for faster computation
                    with autocast_if(self.use_mix_precision):
                        # [num_traj, num_segments, 1 + num_seg_actions]
                        vq_pred = self.critic.critic(
                            net=net, d_state=None, c_state=c_state,
                            actions=seg_actions, idx_d=None, idx_c=seg_start_idx,
                            idx_a=seg_actions_idx)  # Exclude v-func at 0th

                        # Mask out the padded actions
                        # [num_traj, num_segments, num_seg_actions]
                        valid_mask = seg_actions_idx < self.traj_length

                        # [num_traj, num_segments, num_seg_actions]
                        vq_pred[..., 1:] = vq_pred[..., 1:] * valid_mask
                        targets[..., 1:] = targets[..., 1:] * valid_mask

                        # Loss
                        critic_loss = mse(vq_pred[..., 1:], targets[..., 1:])  # Q

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
            elif self.return_type == "segment_n_step_return_vf":
                idx_in_segments = self.get_segments(pad_additional=True)
                seg_start_idx = idx_in_segments[..., 0]
                assert seg_start_idx[-1] < self.traj_length
                seg_actions_idx = idx_in_segments[..., :-1]
                num_seg_actions = seg_actions_idx.shape[-1]

                # [num_traj, num_segments, dim_state]
                c_state = states[:, seg_start_idx]
                seg_start_idx = util.add_expand_dim(seg_start_idx, [0], [num_traj])

                padded_actions = torch.nn.functional.pad(
                    actions, (0, 0, 0, num_seg_actions), "constant", 0)

                # [num_traj, num_segments, num_seg_actions, dim_action]
                seg_actions = padded_actions[:, seg_actions_idx]

                # [num_traj, num_segments, num_seg_actions]
                seg_actions_idx = util.add_expand_dim(seg_actions_idx, [0],
                                                      [num_traj])

                # [num_traj, num_segments, 1 + num_seg_actions]
                targets = self.segments_n_step_return_vf(dataset, idx_in_segments)

                # Log targets and MC returns
                mc_returns_mean = util.compute_mc_return(
                    dataset["step_rewards"].mean(dim=0),
                    self.discount_factor).mean().item()

                mc_returns_list.append(mc_returns_mean)
                targets_mean = targets.mean().item()
                targets_list.append(targets_mean)
                targets_bias_list.append(targets_mean - mc_returns_mean)

                for net, target_net, opt, scaler in self.critic_nets_and_opt():
                    # Use mix precision for faster computation
                    with autocast_if(self.use_mix_precision):
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
                        critic_loss = mse(vq_pred, targets)  # Both V and Q

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
                # Compute targets, [num_traj, traj_length]
                true_returns = self.compute_true_return(dataset)

                # Update Critics, iteratively backward
                for state_idx in reversed(range(traj_length)):
                    c_state = states[:, state_idx]
                    seg_actions = actions[:, state_idx:]
                    idx_a = torch.arange(state_idx, traj_length,
                                         device=self.device)
                    idx_a = util.add_expand_dim(idx_a, [0], [num_traj])
                    idx_c = torch.full([num_traj], state_idx,
                                       device=self.device)

                    # V-func + a sequence of Q-func
                    for net, target_net, opt, scaler in self.critic_nets_and_opt():
                        # Use mix precision for faster computation
                        with autocast_if(self.use_mix_precision):
                            # [num_traj, num_segments, 1 + num_seg_actions]
                            q_pred = self.critic.critic(
                                net=net, d_state=None, c_state=c_state,
                                actions=seg_actions, idx_d=None, idx_c=idx_c,
                                idx_a=idx_a)[..., 1:]  # Exclude v-func at 0th

                            targets = true_returns[..., state_idx][..., None]
                            targets = targets.repeat(1, q_pred.shape[-1])

                            # Loss
                            critic_loss = mse(q_pred, targets)

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

            else:
                raise ValueError("Unknown return type")

        update_critic_time = util.run_time_test(lock=False, key="update critic")
        self.targets_overestimate = np.mean(targets_bias_list) > 1

        # policy update logs
        policy_loss_list = []
        trust_region_loss_list = []
        entropy_loss_list = []
        q_loss_list = []
        trust_region_loss_grad_norm_list = []
        entropy_loss_grad_norm_list = []
        q_loss_grad_norm_list = []
        entropy_list = []
        policy_grad_norm = []
        clipped_policy_grad_norm = []
        update_policy_net_time = []
        projection_time = []
        tr_kl_dict = self.get_dict_for_kl_statistic()

        ########################################################################
        #                             Update policy
        ########################################################################
        util.run_time_test(lock=True, key="update policy")

        for grad_idx in range(self.epochs_policy):
            if not update_policy:
                break

            # Sample from replay buffer
            dataset = self.replay_buffer.sample(self.batch_size,
                                                normalize=self.norm_data,
                                                use_priority=False,
                                                policy_recent=True)

            # Make a new prediction
            pred_mean, pred_L, info = self.make_new_pred(dataset,
                                                         tr_kl_dict=tr_kl_dict)

            if self.projection is not None:
                tr_kl_dict = info["tr_kl_dict"]
                trust_region_loss = info["trust_region_loss"]
                projection_time.append(info["projection_time"])
            else:
                tr_kl_dict = {}
                trust_region_loss = torch.zeros([1], device=self.device)
                projection_time.append(0)

            # Entropy penalty loss
            entropy_loss, entropy_stats = self.entropy_loss(pred_mean, pred_L)

            q_loss = self.q_loss(dataset, pred_mean, pred_L)

            # clone the loss and backward to compute the grad individually?
            if self.log_now:
                q_loss_grad, entropy_loss_grad, trust_region_loss_grad \
                    = util.grad_from_each_loss(self.policy,
                                               self.policy_optimizer,
                                               q_loss, entropy_loss,
                                               trust_region_loss)
            else:
                q_loss_grad = entropy_loss_grad = trust_region_loss_grad = 0.

            policy_loss = q_loss + entropy_loss + trust_region_loss

            # Update policy net parameters
            util.run_time_test(lock=True, key="update policy net")
            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()

            if self.clip_grad_norm > 0 or self.log_now:
                grad_norm, grad_norm_c = util.grad_norm_clip(
                    self.clip_grad_norm, self.policy_net_params)
            else:
                grad_norm, grad_norm_c = 0., 0.

            self.policy_optimizer.step()

            update_policy_net_time.append(
                util.run_time_test(lock=False, key="update policy net"))

            # Logging
            policy_loss_list.append(policy_loss.item())
            trust_region_loss_list.append(trust_region_loss.item())
            entropy_loss_list.append(entropy_loss.item())
            trust_region_loss_grad_norm_list.append(trust_region_loss_grad)
            entropy_loss_grad_norm_list.append(entropy_loss_grad)
            q_loss_grad_norm_list.append(q_loss_grad)
            entropy_list.append(entropy_stats["entropy"])
            q_loss_list.append(q_loss.item())

            policy_grad_norm.append(grad_norm)
            clipped_policy_grad_norm.append(grad_norm_c)
        if self.use_old_policy:
            self.policy_old.copy_(self.policy, self.old_policy_update_rate)
        update_policy_time = util.run_time_test(lock=False, key="update policy")
        ######################### End of gradient step #########################

        if self.log_now:
            # Get critic update statistics
            critic_info_dict = {
                "update_critic_time": update_critic_time,
                "update_critic_net_time": math.fsum(update_critic_net_time),
                "update_target_net_time": math.fsum(update_target_net_time),
                **util.generate_stats(critic_loss_list, "critic_loss"),
                **util.generate_stats(critic_grad_norm, "critic_grad_norm"),
                **util.generate_stats(clipped_critic_grad_norm,
                                      "clipped_critic_grad_norm"),
                **util.generate_stats(mc_returns_list, "mc_returns"),
                **util.generate_stats(targets_list, "targets"),
                **util.generate_stats(targets_bias_list, "targets_bias")
            }

            # Get policy update statistics
            if update_policy:
                policy_info_dict = {
                    "update_policy_time": update_policy_time,
                    "update_policy_net_time": math.fsum(update_policy_net_time),
                    **util.generate_stats(q_loss_list, "q_loss_raw"),
                    **util.generate_stats(entropy_loss_list, "entropy_loss"),
                    **util.generate_stats(trust_region_loss_list,
                                          "trust_region_loss"),
                    **util.generate_stats(q_loss_grad_norm_list,
                                          "q_loss_grad_norm"),
                    **util.generate_stats(entropy_loss_grad_norm_list,
                                          "entropy_loss_grad_norm"),
                    **util.generate_stats(trust_region_loss_grad_norm_list,
                                          "trust_region_loss_grad_norm"),
                    **util.generate_stats(policy_loss_list, "policy_loss"),
                    **util.generate_stats(entropy_list, "entropy"),
                    **util.generate_stats(policy_grad_norm, "policy_grad_norm"),
                    **util.generate_stats(clipped_policy_grad_norm,
                                          "clipped_policy_grad_norm"),
                    **util.generate_many_stats(tr_kl_dict, "projection"),
                    "projection_time": np.sum(projection_time)
                }
            else:
                policy_info_dict = {}
        else:
            # do not log anything
            critic_info_dict = {}
            policy_info_dict = {}

        return critic_info_dict, policy_info_dict

    def q_loss(self, dataset, pred_mean, pred_L):
        self.critic.eval()  # disable dropout
        self.critic.requires_grad(False)

        states = dataset["step_states"]
        init_time = dataset["segment_init_time"]
        init_pos = dataset["segment_init_pos"]
        init_vel = dataset["segment_init_vel"]
        decision_idx = dataset["decision_idx"]
        num_traj = states.shape[0]
        times = self.sampler.get_times(init_time, self.sampler.num_times,
                                       self.traj_has_downsample)

        # Shape of idx_in_segments [num_segments, num_seg_actions + 1]
        idx_in_segments = self.get_segments()
        num_segments = idx_in_segments.shape[0]
        seg_start_idx = idx_in_segments[..., 0]
        seg_actions_idx = idx_in_segments[..., :-1]

        # Get the trajectory segments
        # [num_trajs, num_segments, num_seg_actions, num_dof]
        # Note, here the init condition of the traj is used by all segments
        pred_seg_actions = self.policy.sample_segments(
            require_grad=True, params_mean=pred_mean,
            params_L=pred_L, times=times, init_time=init_time,
            init_pos=init_pos, init_vel=init_vel,
            idx_in_segments=idx_in_segments, use_mean=False, pos_only=False)

        # Normalize actions
        if self.norm_data:
            pred_seg_actions = self.replay_buffer.normalize_data(
                "step_actions", pred_seg_actions)

        # Decision state
        # [num_traj, dim_state]
        d_state = states[
            torch.arange(num_traj, device=self.device), decision_idx]

        # Decision state
        # -> [num_trajs, num_segments, dim_state]
        d_state = util.add_expand_dim(d_state, [1],
                                      [num_segments])

        # Current state
        # [num_traj, num_segments, dim_state]
        c_state = states[:, seg_start_idx]

        c_net1 = self.critic.net1
        c_net2 = self.critic.net2

        # [num_traj] -> [num_traj, num_segments]
        decision_idx = util.add_expand_dim(decision_idx, [1],
                                           [num_segments])
        # [num_segments] -> [num_traj, num_segments]
        seg_start_idx = util.add_expand_dim(seg_start_idx, [0],
                                            [num_traj])
        # [num_segments, num_actions] -> [num_traj, num_segments, num_actions]
        seg_actions_idx = util.add_expand_dim(seg_actions_idx, [0],
                                              [num_traj])

        # [num_traj, num_segments, num_seg_actions]
        # vq -> q
        q1 = self.critic.critic(net=c_net1, d_state=None, c_state=c_state,
                                actions=pred_seg_actions, idx_d=None,
                                idx_c=seg_start_idx,
                                idx_a=seg_actions_idx)[..., 1:]
        if not self.critic.single_q:
            q2 = self.critic.critic(net=c_net2, d_state=None, c_state=c_state,
                                    actions=pred_seg_actions, idx_d=None,
                                    idx_c=seg_start_idx,
                                    idx_a=seg_actions_idx)[..., 1:]
        else:
            q2 = q1

        if self.type_policy_q_update == "mean_of_segment_mini_sum_of_actions":
            # Sum over actions in each segment
            # [num_traj, num_segments, num_seg_actions]
            # -> [num_traj, num_segments]
            q1 = q1.sum(dim=-1)
            q2 = q2.sum(dim=-1)
            # Minimum over q1, q2, [num_traj, num_segments]
            q = torch.minimum(q1, q2)
            # Mean over trajs and segments
            # [num_traj, num_segments] -> scalar
            q_loss = -q.mean()

        elif self.type_policy_q_update == "sum_of_segment_mini_sum_of_actions":
            # Sum over segment actions
            # [num_traj, num_segments, num_seg_actions]
            # -> [num_traj, num_segments]
            q1 = q1.sum(dim=-1)
            q2 = q2.sum(dim=-1)
            # Minimum over q1, q2, [num_traj, num_segments]
            q = torch.minimum(q1, q2)
            # Sum over segments, [num_traj, num_segments] -> [num_traj]
            q = q.sum(dim=-1)
            # Mean over trajs, [num_traj] -> scalar
            q_loss = -q.mean()

        elif self.type_policy_q_update == "mean_of_segment_sum_of_mini_actions":
            # Minimum over q1, q2, [num_traj, num_segments, num_seg_actions]
            q = torch.minimum(q1, q2)
            # Sum over segment actions and segments
            # [num_traj, num_segments, num_seg_actions] -> [num_traj, num_segs]
            q = q.sum(dim=[-1])
            # Mean over trajs and segments, [num_traj, num_segs] -> scalar
            q_loss = -q.mean()

        elif self.type_policy_q_update == "sum_of_segment_sum_of_mini_actions":
            # Minimum over q1, q2, [num_traj, num_segments, num_seg_actions]
            q = torch.minimum(q1, q2)
            # Sum over segment actions and segments
            # [num_traj, num_segments, num_seg_actions] -> [num_traj]
            q = q.sum(dim=[-2, -1])
            # Mean over trajs, [num_traj] -> scalar
            q_loss = -q.mean()

        elif self.type_policy_q_update == "mini_mean_of_segment_sum_of_actions":
            # Sum over actions in each segment
            # [num_traj, num_segments, num_seg_actions]
            # -> [num_traj, num_segments]
            # -> [num_traj]
            q1 = q1.sum(dim=-1).mean(dim=-1)
            q2 = q2.sum(dim=-1).mean(dim=-1)
            # Minimum over q1, q2, [num_traj]
            q = torch.minimum(q1, q2)
            # Mean over trajs, [num_traj] -> scalar
            q_loss = -q.mean()
        elif self.type_policy_q_update == "mini_sum_of_segment_sum_of_actions":
            # Sum over actions in each segment
            # [num_traj, num_segments, num_seg_actions]
            # -> [num_traj, num_segments]
            # -> [num_traj]
            q1 = q1.sum(dim=[-2, -1])
            q2 = q2.sum(dim=[-2, -1])
            # Minimum over q1, q2, [num_traj]
            q = torch.minimum(q1, q2)
            # Mean over trajs, [num_traj] -> scalar
            q_loss = -q.mean()

        elif self.type_policy_q_update == "mean_of_segment_mini_last_action":
            # Sum over actions in each segment
            # [num_traj, num_segments, num_seg_actions]
            # -> [num_traj, num_segments]
            q1 = q1[..., -1]
            q2 = q2[..., -1]
            # Minimum over q1, q2, [num_traj, num_segments]
            q = torch.minimum(q1, q2)
            # Mean over trajs and segments
            # [num_traj, num_segments] -> scalar
            q_loss = -q.mean()

        else:
            raise ValueError("Unknown type_policy_q_update")

        return q_loss

    def kl_old_new_proj(self, params_mean_new, params_L_new,
                        params_mean_old, params_L_old,
                        proj_mean, proj_L,
                        result_dict):
        # KL(new || old)
        (new_old_mean_diff, new_old_cov_diff,
         new_old_shape_diff, new_old_volume_diff) = \
            gaussian_kl_details(self.policy,
                                (params_mean_new, params_L_new),
                                (params_mean_old, params_L_old))

        # KL(new || proj)
        (new_proj_mean_diff, new_proj_cov_diff,
         new_proj_shape_diff, new_proj_volume_diff) = \
            gaussian_kl_details(self.policy, (params_mean_new, params_L_new),
                                (proj_mean, proj_L))

        # KL(proj || old)
        (proj_old_mean_diff, proj_old_cov_diff,
         proj_old_shape_diff, proj_old_volume_diff) = \
            gaussian_kl_details(self.policy, (proj_mean, proj_L),
                                (params_mean_old, params_L_old))
        result_dict["new_old_mean_diff"].append(
            util.to_np(new_old_mean_diff.mean()))
        result_dict["new_old_cov_diff"].append(
            util.to_np(new_old_cov_diff.mean()))
        result_dict["new_old_shape_diff"].append(
            util.to_np(new_old_shape_diff.mean()))
        result_dict["new_old_volume_diff"].append(
            util.to_np(new_old_volume_diff.mean()))
        result_dict["new_proj_mean_diff"].append(
            util.to_np(new_proj_mean_diff.mean()))
        result_dict["new_proj_cov_diff"].append(
            util.to_np(new_proj_cov_diff.mean()))
        result_dict["new_proj_shape_diff"].append(
            util.to_np(new_proj_shape_diff.mean()))
        result_dict["new_proj_volume_diff"].append(
            util.to_np(new_proj_volume_diff.mean()))
        result_dict["proj_old_mean_diff"].append(
            util.to_np(proj_old_mean_diff.mean()))
        result_dict["proj_old_cov_diff"].append(
            util.to_np(proj_old_cov_diff.mean()))
        result_dict["proj_old_shape_diff"].append(
            util.to_np(proj_old_shape_diff.mean()))
        result_dict["proj_old_volume_diff"].append(
            util.to_np(proj_old_volume_diff.mean()))
        return result_dict

    def value_loss(self, values: torch.Tensor, returns: torch.Tensor,
                   old_vs: torch.Tensor):
        """
        Adapt from TROL
        Computes the value function loss.

        When using GAE we have L_t = ((v_t + A_t).detach() - v_{t})
        Without GAE we get L_t = (r(s,a) + y*V(s_t+1) - v_{t}) accordingly.

        Optionally, we clip the value function around the original value of v_t

        Returns:
        Args:
            values: value estimates
            returns: computed returns with GAE or n-step
            old_vs: old value function estimates from behavior policy

        Returns:
            Value function loss
        """

        vf_loss = (returns - values).pow(2)

        if self.clip_critic > 0:
            vs_clipped = old_vs + (values - old_vs).clamp(-self.clip_critic,
                                                          self.clip_critic)
            vf_loss_clipped = (vs_clipped - returns).pow(2)
            vf_loss = torch.max(vf_loss, vf_loss_clipped)
        return vf_loss.mean()

    @staticmethod
    def surrogate_loss(advantages, log_prob_new, log_prob_old):
        """
        Computes the surrogate reward for Importance Sampling policy gradient

        Args:
            advantages: advantages
            log_prob_new: the log probabilities from current policy
            log_prob_old: the log probabilities

        Returns:
            surrogate_loss: the surrogate loss
            stats_dict: statistics
        """

        ratio = (log_prob_new - log_prob_old).exp()

        advantage_weighted_log_prob = ratio * advantages
        surrogate_loss = advantage_weighted_log_prob.mean()

        stats_dict = {"imp_smp_ratio": ratio.mean().item()}
        return -surrogate_loss, stats_dict

    def entropy_loss(self, params_mean, params_L):
        entropy = self.policy.entropy([params_mean, params_L]).mean()
        entropy_loss = -self.entropy_penalty_coef * entropy
        stats_dict = {"entropy": entropy.item()}
        return entropy_loss, stats_dict

    @staticmethod
    def get_dict_for_kl_statistic():
        # Trust Region KL storage
        tr_kl_dict = {"new_old_mean_diff": [],
                      "new_old_cov_diff": [],
                      "new_old_shape_diff": [],
                      "new_old_volume_diff": [],
                      "new_proj_mean_diff": [],
                      "new_proj_cov_diff": [],
                      "new_proj_shape_diff": [],
                      "new_proj_volume_diff": [],
                      "proj_old_mean_diff": [],
                      "proj_old_cov_diff": [],
                      "proj_old_shape_diff": [],
                      "proj_old_volume_diff": []}
        return tr_kl_dict

    def fix_relative_goal_for_segments(self,
                                       params: torch.Tensor,
                                       traj_init_pos: torch.Tensor,
                                       segments_init_pos: torch.Tensor):
        """

        Args:
            params: ProDMP parameters, [*add_dim, num_segments, num_weights]
            traj_init_pos: [*add_dim, num_dof]
            segments_init_pos: [*add_dim, num_segments, num_dof]

        Returns:

        """
        relative_goal = self.policy.mp.relative_goal

        if relative_goal:
            # [*add_dim, num_segments, num_dof]
            delta_relative_goal \
                = segments_init_pos - traj_init_pos[..., None, :]
            num_basis_g = self.policy.mp.num_basis_g

            # As, abs_goal = rel_goal + traj_init_pos
            # Set: delta = seg_init_pos - traj_init_pos
            # -> traj_init_pos = seg_init_pos - delta
            # So: abs_goal = rel_goal + seg_init_pos - delta
            #              = fix_rel_goal + seg_init_pos
            # So, fix_rel_goal = rel_goal - delta

            params = params.clone()
            # [*add_dim, num_segments, num_dof]
            params[..., num_basis_g - 1::num_basis_g] \
                = (params[..., num_basis_g - 1::num_basis_g] - delta_relative_goal)

            return params

        else:
            return params

    def critic_nets_and_opt(self):
        if self.critic.single_q:
            return zip(util.make_iterable(self.critic.net1, "list"),
                       util.make_iterable(self.critic.target_net1, "list"),
                       util.make_iterable(self.critic_optimizer[0], "list"),
                       util.make_iterable(self.critic_grad_scaler[0], "list"))
        else:
            return zip([self.critic.net1, self.critic.net2],
                       [self.critic.target_net1, self.critic.target_net2],
                       self.critic_optimizer, self.critic_grad_scaler)

    @staticmethod
    def normalize_q(q):
        with torch.no_grad():
            q_mean = q.mean()
            q_std = q.std()
        normalized_q = (q - q_mean) / (q_std + 1e-8)
        return normalized_q

    def save_agent(self, log_dir: str, epoch: int):
        super().save_agent(log_dir, epoch)
        self.sampler.save_rms(log_dir, epoch)

    def load_agent(self, log_dir: str, epoch: int):
        super().load_agent(log_dir, epoch)
        self.sampler.load_rms(log_dir, epoch)
        self.fresh_agent = False
