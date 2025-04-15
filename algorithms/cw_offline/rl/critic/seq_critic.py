import copy

import torch.nn.functional

from algorithms.cw_offline import util
from algorithms.cw_offline.rl.critic import AbstractCritic
from algorithms.cw_offline.util import TrajectoryQfunctionGPT


class SeqCritic(AbstractCritic):
    def __init__(self, **config):
        self.single_q = config.get("single_q", False)
        self.config = config
        self.dtype, self.device = util.parse_dtype_device(config["dtype"],
                                                          config["device"])
        self.net1 = None
        self.net2 = None
        self.target_net1 = None
        self.target_net2 = None
        self.eta = config["update_rate"]
        self._create_network()

    def _create_network(self):
        """
        Create critic net with given configuration

        @return:
            None
        """
        # check how SAC initializes the critic
        config1 = copy.deepcopy(self.config)
        config1["name"] = self._critic_net_type + "_1"
        config2 = copy.deepcopy(self.config)
        config2["name"] = self._critic_net_type + "_2"

        self.net1 = TrajectoryQfunctionGPT(**config1)
        self.net1.train()
        self.target_net1 = copy.deepcopy(self.net1)
        self.target_net1.requires_grad_(False)
        self.target_net1.eval()

        if not self.single_q:
            self.net2 = TrajectoryQfunctionGPT(**config2)
            self.net2.train()
            self.target_net2 = copy.deepcopy(self.net2)
            self.target_net2.requires_grad_(False)
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

    def eval(self):
        self.net1.eval()
        if not self.single_q:
            self.net2.eval()

    def train(self):
        self.net1.train()
        if not self.single_q:
            self.net2.train()

    @property
    def parameters(self) -> []:
        """
        Get network parameters
        Returns:
            parameters
        """
        if self.single_q:
            raise NotImplementedError("Unclear behaviour of single Q")
        return [self.net1.parameters(), self.net2.parameters()]

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

    def update_target_net(self, net, target_net):
        if self.single_q:
            assert self.net1 == self.net2
            assert self.target_net1 == self.target_net2

        for target_param, source_param in zip(target_net.parameters(),
                                              net.parameters()):
            target_param.data.copy_(self.eta * source_param.data
                                    + (1 - self.eta) * target_param.data)

    def requires_grad(self, requires_grad):
        self.net1.requires_grad_(requires_grad)
        if not self.single_q:
            self.net2.requires_grad_(requires_grad)
