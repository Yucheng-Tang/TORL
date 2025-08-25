import os
from cw2 import cw_error
from cw2 import experiment
from cw2.cw_data import cw_logging
from tqdm import tqdm
import algorithms.cw_offline.util as util
from algorithms.cw_offline.rl.agent import agent_factory
from algorithms.cw_offline.rl.critic import critic_factory
from algorithms.cw_offline.rl.policy import policy_factory
from algorithms.cw_offline.rl.projection import projection_factory
from algorithms.cw_offline.rl.sampler import sampler_factory
from algorithms.cw_offline.rl.replay_buffer import replay_buffer_factory
import psutil
import copy
import time

import gym
import d4rl

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union


class MPExperimentSingleProcess(experiment.AbstractIterativeExperiment):
    def initialize(self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        cfg = cw_config["params"]
        cpu_cores = cw_config.get("cpu_cores", set(range(psutil.cpu_count(logical=True))))
        util.set_global_random_seed(cw_config["seed"])
        self.verbose_level = cw_config.get("verbose_level", 1)

        load_model_dir = cw_config.get('load_model_dir', None)
        load_model_epoch = cw_config.get('load_model_epoch', None)

        self.training = load_model_dir is None or cw_config["keep_training"]

        self.save_model_dir = (os.path.abspath(cw_config["save_model_dir"]) if self.training and cw_config.get("save_model_dir", None) else None)
        self.save_model_interval = max(cw_config["iterations"] // cw_config["num_checkpoints"], 1) if self.save_model_dir else None

        # self.sampler = sampler_factory(cfg["sampler"]["type"], cpu_cores=cpu_cores, disable_train_env=not self.training, **cfg["sampler"]["args"])
        # state_dim = self.get_dim_in(cfg, self.sampler)
        # policy_out_dim = self.dim_policy_out(cfg)

        self.env_name = cw_config.get('env_name', None)
        self.env = gym.make(self.env_name)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        dataset = d4rl.qlearning_dataset(self.env)


        self.normalized_reward = cw_config.get('normalized_reward', True) # TODO: add into the config

        if self.normalize_reward:
            # modify_reward(
            #     dataset,
            #     config.env,
            #     reward_scale=config.reward_scale,
            #     reward_bias=config.reward_bias,
            # )

            self.modify_reward(
                dataset,
                self.env_name,
                reward_scale=self.reward_scale,
                reward_bias=self.reward_bias,
            )
        if self.normalize:
            state_mean, state_std = self.compute_mean_std(dataset["observations"], eps=1e-3)
        else:
            state_mean, state_std = 0, 1

        self.policy = policy_factory(cfg["policy"]["type"], dim_in=state_dim, dim_out=action_dim, **cfg["policy"]["args"])
        # action_dim = self.policy.num_dof * 2

        self.critic = critic_factory(cfg["critic"]["type"], state_dim=state_dim, action_dim=action_dim, **cfg["critic"]["args"])

        # self.projection = projection_factory(cfg["projection"]["type"], action_dim=policy_out_dim, **cfg["projection"]["args"])

        replay_buffer_data_shape = {
            "observations": (state_dim,),
            "actions": (action_dim,),
            "rewards": (1,),
            "next_observations": (state_dim,),
            "terminals": (1,),
        }
        # Data normalization is done in dataset generation
        norm_obs = cfg["agent"]["args"].get("norm_obs", False)
        replay_buffer_norm_info = {key: key in ["observations", "next_observations"] for key in replay_buffer_data_shape} if norm_obs else None

        # replay_buffer_norm_info = {
        #     "observations": False,  # True,
        #     "actions": False,
        #     "rewards": False,
        #     "next_observations": False,  # True,
        #     "terminals": False,
        # }

        self.replay_buffer = replay_buffer_factory(cfg["replay_buffer"]["type"], data_info=replay_buffer_data_shape, data_norm_info=replay_buffer_norm_info, **cfg["replay_buffer"]["args"])

        traj_length = self.replay_buffer.sequence_length

        self.agent = agent_factory(cfg["agent"]["type"], policy=self.policy, critic=self.critic, conn=None, replay_buffer=self.replay_buffer, traj_length=traj_length, **cfg["agent"]["args"])

        if load_model_dir:
            self.agent.load_agent(load_model_dir, load_model_epoch)
            util.print_line_title("Testing")
        else:
            util.print_line_title("Training")

        self.progress_bar = tqdm(total=cw_config["iterations"])
        self.exp_start_time = time.perf_counter()

    def iterate(self, cw_config: dict, rep: int, n: int) -> dict:
        if self.training:
            result_metrics = self.agent.step()
            result_metrics["exp_speed"] = self.experiment_speed(n)
            self.progress_bar.update(1)
            return self._filter_metrics(result_metrics)
        else:
            deterministic_result_dict, _ = self.agent.evaluate(render=True)
            self.progress_bar.update(1)
            return deterministic_result_dict

    def _filter_metrics(self, metrics):
        if self.verbose_level == 0:
            return {}
        elif self.verbose_level == 1:
            return {k: v for k, v in metrics.items() if not any(skip in k for skip in ["exploration", "projection", "gradient", "grad_norm", "clipped_", "mc_returns", "targets_bias", "step_", "time_limit_dones", "segment", "update_", "median", "targets", "entropy", "trust_region", "loss", "critic"])}
        return metrics

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        if self.save_model_dir and ((n + 1) % self.save_model_interval == 0 or (n + 1) == cw_config["iterations"]):
            self.agent.save_agent(log_dir=self.save_model_dir, epoch=n + 1)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass

    def experiment_speed(self, n):
        return (time.perf_counter() - self.exp_start_time) / (n + 1)

    def modify_reward(
            self,
            dataset: Dict,
            env_name: str,
            max_episode_steps: int = 1000,
            reward_scale: float = 1.0,
            reward_bias: float = 0.0,
    ):
        if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
            min_ret, max_ret = self.return_reward_range(dataset, max_episode_steps)
            dataset["rewards"] /= max_ret - min_ret
            dataset["rewards"] *= max_episode_steps
        dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias

    @staticmethod
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

    @staticmethod
    def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
        mean = states.mean(0)
        std = states.std(0) + eps
        return mean, std


if __name__ == "__main__":
    for key in os.environ.keys():
        if "-xCORE-AVX2" in os.environ[key]:
            os.environ[key] = os.environ[key].replace("-xCORE-AVX2", "")

    util.RLExperiment(MPExperimentSingleProcess, True)