from cw2 import experiment
from cw2.cw_data import cw_logging
# from ..offline.cql import TrainConfig  # use your function directly or repackage

import os
from cw2 import cw_error
from cw2 import experiment
from cw2.cw_data import cw_logging
from tqdm import tqdm

import algorithms.cw_offline.util as util
from algorithms.cw_offline.rl.agent import agent_factory
from algorithms.cw_offline.rl.critic import critic_factory
from algorithms.cw_offline.rl.policy import policy_factory
# from algorithms.cw_offline.rl.projection import projection_factory
# from algorithms.cw_offline.rl.sampler import sampler_factory
from algorithms.cw_offline.rl.replay_buffer import replay_buffer_factory
import psutil
import copy
import time

import d4rl
import gym

class CQLCW2Experiment(experiment.AbstractIterativeExperiment):
    def initialize(self,
                   cw_config: dict,
                   rep: int,
                   logger: cw_logging.LoggerArray):
        # Get experiment parameters
        cfg = cw_config["params"]
        util.set_global_random_seed(cw_config["seed"])
        self.verbose_level = cw_config.get("verbose_level", 1)

        # Determine training or testing mode
        load_model_dir = cw_config.get('load_model_dir', None)
        load_model_epoch = cw_config.get('load_model_epoch', None)

        if load_model_dir is None or cw_config["keep_training"]:
            self.training = True
        else:
            self.training = False

        if self.training and cw_config.get("save_model_dir", None) is not None:
            # Save model in training mode
            self.save_model_dir = os.path.abspath(cw_config["save_model_dir"])
            self.save_model_interval = \
                max(cw_config["iterations"] // cw_config["num_checkpoints"], 1)

        else:
            # In testing mode or no save model dir in training mode
            self.save_model_dir = None
            self.save_model_interval = None

        # Load environment and dataset
        env = gym.make(cw_config["env"])

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        dataset = d4rl.qlearning_dataset(env)

        # Components


        replay_buffer_data_shape = {
            "observations": (state_dim, ),
            "actions": (action_dim,),
            "rewards": (1,),
            "next_observations": (state_dim,),
            "terminals": (1,),
        }
        if cfg["agent"]["args"].get("norm_obs", False):
            replay_buffer_norm_info = {
                "observations": True,
                "actions": False,
                "rewards": False,
                "next_observations": True,
                "terminals": False,
            }
        else:
            replay_buffer_norm_info = None

        # Initialize replay buffer
        self.replay_buffer = replay_buffer_factory(
            cfg["replay_buffer"]["type"],
            data_info=replay_buffer_data_shape,
            data_norm_info=replay_buffer_norm_info,
            **cfg["replay_buffer"]["args"]
            # buffer_size=2000000,
            # dtype="float32",
            # device="cuda",
        )

        # policy_out_dim = self.dim_policy_out(cfg)
        max_action = float(env.action_space.high[0])
        self.policy = policy_factory(cfg["policy"]["type"],
                                     dim_in=state_dim,
                                     dim_out=action_dim,  # policy_out_dim,
                                     max_action=max_action,

                                     **cfg["policy"]["args"])

        self.critic = critic_factory(cfg["critic"]["type"],
                                     state_dim=state_dim,
                                     action_dim=action_dim,
                                     **cfg["critic"]["args"])

        self.agent = agent_factory(cfg["agent"]["type"],
                                   policy=self.policy,
                                   critic=self.critic,
                                   sampler=None,
                                   projection=None,
                                   replay_buffer=self.replay_buffer,
                                   **cfg["agent"]["args"])

        # Load model if it in testing mode
        if load_model_dir is None:
            util.print_line_title("Training")
        else:
            self.agent.load_agent(load_model_dir, load_model_epoch)
            util.print_line_title("Testing")

        # self.cfg = TrainConfig(**params)
        # self.repetitions = rep
        # self.logger = logger
        # self._step = 0
        #
        # # Set up environment, dataset, replay buffer, and CQL agent just like your `train()` function
        # self.env = gym.make(self.cfg.env)
        # ...
        # self.trainer = ContinuousCQL(...)  # pass your config
        # self.actor = self.trainer.actor

    def iterate(self, cw_config, rep, n):
        # # Sample batch from the replay buffer
        # batch = self.replay_buffer.sample(self.cfg.batch_size)
        # batch = [b.to(self.cfg.device) for b in batch]
        # log_dict = self.trainer.train(batch)
        # self.logger.log(log_dict)
        #
        # self._step += 1
        # if self._step % self.cfg.eval_freq == 0:
        #     eval_score = eval_actor(
        #         self.env,
        #         self.actor,
        #         device=self.cfg.device,
        #         n_episodes=self.cfg.n_episodes,
        #         seed=self.cfg.seed,
        #     )
        #     score = eval_score.mean()
        #     norm_score = self.env.get_normalized_score(score) * 100
        #     self.logger.log({
        #         "eval_score": score,
        #         "d4rl_normalized_score": norm_score
        #     })
        return {}

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        if self.save_model_dir and ((n + 1) % self.save_model_interval == 0
                                    or (n + 1) == cw_config["iterations"]):
            self.agent.save_agent(log_dir=self.save_model_dir, epoch=n + 1)

    def finalize(self, *args, **kwargs):
        print("Training complete")


def main():
    util.RLExperiment(CQLCW2Experiment, True)
    # exp = CQLCW2Experiment()
    # runner.run(experiment=exp, config_path="configs/cql_halfcheetah.yaml")

if __name__ == "__main__":
    print("Hello World, this is CQL CW2 Experiment!")
    util.RLExperiment(CQLCW2Experiment, True)
    # main()
