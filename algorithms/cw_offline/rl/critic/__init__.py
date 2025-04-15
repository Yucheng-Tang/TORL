from typing import Literal

from .abstract_critic import *
from .regularized_q_critic import *
# from .value_function_critic import *
from .seq_critic import *

def critic_factory(typ: Literal["ValueFunction"],
                   **kwargs):
    """
    Factory methods to instantiate a critic
    Args:
        typ: critic class type
        **kwargs: keyword arguments

    Returns:

    """
    return eval(typ + "(**kwargs)")