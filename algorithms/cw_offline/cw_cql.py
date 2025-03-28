from cw2 import experiment
from cw2.cw_data import cw_logging
from ..offline.cql import TrainConfig  # use your function directly or repackage

import os
from cw2 import cw_error
from cw2 import experiment
from cw2.cw_data import cw_logging
from tqdm import tqdm

# import mprl.util as util
# from mprl.rl.agent import agent_factory
# from mprl.rl.critic import critic_factory
# from mprl.rl.policy import policy_factory
# from mprl.rl.projection import projection_factory
# from mprl.rl.sampler import sampler_factory
# from mprl.rl.replay_buffer import replay_buffer_factory
import psutil
import copy
import time

class CQLCW2Experiment(experiment.AbstractIterativeExperiment):
    def initialize(self,
                   cw_config: dict,
                   rep: int,
                   logger: cw_logging.LoggerArray):
        # Get experiment parameters
        params = cw_config["params"]
        self.cfg = TrainConfig(**params)
        self.repetitions = rep
        self.logger = logger
        self._step = 0

        # Set up environment, dataset, replay buffer, and CQL agent just like your `train()` function
        self.env = gym.make(self.cfg.env)
        ...
        self.trainer = ContinuousCQL(...)  # pass your config
        self.actor = self.trainer.actor

    def iterate(self, cw_config, rep, n):
        # Sample batch from the replay buffer
        batch = self.replay_buffer.sample(self.cfg.batch_size)
        batch = [b.to(self.cfg.device) for b in batch]
        log_dict = self.trainer.train(batch)
        self.logger.log(log_dict)

        self._step += 1
        if self._step % self.cfg.eval_freq == 0:
            eval_score = eval_actor(
                self.env,
                self.actor,
                device=self.cfg.device,
                n_episodes=self.cfg.n_episodes,
                seed=self.cfg.seed,
            )
            score = eval_score.mean()
            norm_score = self.env.get_normalized_score(score) * 100
            self.logger.log({
                "eval_score": score,
                "d4rl_normalized_score": norm_score
            })
        return {}

    def finalize(self, *args, **kwargs):
        print("Training complete")


def main():
    exp = CQLCW2Experiment()
    runner.run(experiment=exp, config_path="configs/cql_halfcheetah.yaml")

if __name__ == "__main__":
    main()