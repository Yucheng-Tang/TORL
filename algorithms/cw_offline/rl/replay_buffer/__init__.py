from typing import Literal
from .seq_replay_buffer import *


def replay_buffer_factory(typ: Literal["DRPReplayBuffer", "ReplayBuffer"],
                          **kwargs):
    """
    Factory methods to instantiate a replay buffer
    Args:
        typ: replay buffer class type
        **kwargs: keyword arguments

    Returns:

    """
    return eval(typ + "(**kwargs)")