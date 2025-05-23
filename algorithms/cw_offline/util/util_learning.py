"""
    Utilities of learning operation
"""
from typing import Union

import numpy as np
import torch

import algorithms.cw_offline.util as util


def joint_to_conditional(joint_mean: Union[np.ndarray, torch.Tensor],
                         joint_L: Union[np.ndarray, torch.Tensor],
                         sample_x: Union[np.ndarray, torch.Tensor]) -> \
        [Union[np.ndarray, torch.Tensor]]:
    """
    Given joint distribution p(x,y), and a sample of x, do:
    Compute conditional distribution p(y|x)
    Args:
        joint_mean: mean of joint distribution
        joint_L: cholesky distribution of joint distribution
        sample_x: samples of x

    Returns:
        conditional mean and L
    """

    # Shape of joint_mean:
    # [*add_dim, dim_x + dim_y]
    #
    # Shape of joint_L:
    # [*add_dim, dim_x + dim_y, dim_x + dim_y]
    #
    # Shape of sample_x:
    # [*add_dim, dim_x]
    #
    # Shape of conditional_mean:
    # [*add_dim, dim_y]
    #
    # Shape of conditional_cov:
    # [*add_dim, dim_y, dim_y]

    # Check dimension
    dim_x = sample_x.shape[-1]
    # dim_y = joint_mean.shape[-1] - dim_x

    # Decompose joint distribution parameters
    mu_x = joint_mean[..., :dim_x]
    mu_y = joint_mean[..., dim_x:]

    L_x = joint_L[..., :dim_x, :dim_x]
    L_y = joint_L[..., dim_x:, dim_x:]
    L_x_y = joint_L[..., dim_x:, :dim_x]

    if util.is_ts(joint_mean):
        cond_mean = mu_y + \
                    torch.einsum('...ik,...lk,...lm,...m->...i', L_x_y, L_x,
                                 torch.cholesky_inverse(L_x), sample_x - mu_x)
    elif util.is_np(joint_mean):
        # Scipy cho_solve does not support batch operation
        cond_mean = mu_y + \
                    np.einsum('...ik,...lk,...lm,...m->...i', L_x_y, L_x,
                              torch.cholesky_inverse(torch.from_numpy(
                                  L_x)).numpy(),
                              sample_x - mu_x)
    else:
        raise NotImplementedError

    cond_L = L_y

    return cond_mean, cond_L


def select_ctx_pred_pts(**kwargs):
    """
    Generate context and prediction indices
    Args:
        **kwargs: keyword arguments

    Returns:
        context indices and prediction indices

    """
    num_ctx = kwargs.get("num_ctx", None)
    num_ctx_min = kwargs.get("num_ctx_min", None)
    num_ctx_max = kwargs.get("num_ctx_max", None)
    first_index = kwargs.get("first_index", None)
    fixed_interval = kwargs.get("fixed_interval", False)
    num_all = kwargs.get("num_all", None)
    num_select = kwargs.get("num_select", None)
    ctx_before_pred = kwargs.get("ctx_before_pred", False)

    # Determine how many points shall be selected
    if num_select is None:
        assert fixed_interval is False
        assert first_index is None
        num_select = num_all
    else:
        assert num_select <= num_all

    # Determine how many context points shall be selected
    if num_ctx is None:
        num_ctx = torch.randint(low=num_ctx_min, high=num_ctx_max, size=(1,))
    assert num_ctx < num_select

    # Select points
    if fixed_interval:
        # Select using fixed interval
        interval = num_all // num_select
        residual = num_all % num_select

        if first_index is None:
            # Determine the first index
            first_index = \
                torch.randint(low=0, high=interval + residual, size=[]).item()
        else:
            # The first index is specified
            assert 0 <= first_index < interval + residual
        selected_indices = torch.arange(start=first_index, end=num_all,
                                        step=interval, dtype=torch.long)
    else:
        # Select randomly
        permuted_indices = torch.randperm(n=num_all)
        selected_indices = torch.sort(permuted_indices[:num_select])[0]

    # split ctx and pred
    if num_ctx == 0:
        # No context
        ctx_idx = []
        pred_idx = selected_indices

    else:
        # Ctx + Pred
        if ctx_before_pred:
            ctx_idx = selected_indices[:num_ctx]
            pred_idx = selected_indices[num_ctx:]
        else:
            permuted_select_indices = torch.randperm(n=num_select)
            ctx_idx = selected_indices[permuted_select_indices[:num_ctx]]
            pred_idx = selected_indices[permuted_select_indices[num_ctx:]]
    return ctx_idx, pred_idx


def select_pred_pairs(**kwargs):
    pred_index = select_ctx_pred_pts(num_ctx=0, **kwargs)[1]
    # pred_pairs = torch.combinations(pred_index, 2)
    pred_pairs = torch.zeros([pred_index.shape[0] - 1, 2])
    pred_pairs[:, 0] = pred_index[:-1]
    pred_pairs[:, 1] = pred_index[1:]
    return pred_pairs


def get_advantage_return(rewards, values, dones, time_limit_dones,
                         discount_factor, use_gae, gae_scaling):
    """
    Directly adapt from TRPL code
    GAE style advantages and return computation
    Args:

        rewards: dataset rewards
        values: estimated values
        dones: flags of termination
        time_limit_dones: flags of reaching max horizon
        gae_scaling: lambda for GAE
        use_gae: use GAE return or MC return
        discount_factor: discount factor

        Shape of rewards:
        [num_env, num_times]

        Shape of values:
        [num_env, num_times + 1]

        Shape of dones:
        [num_env, num_times], dtype = bool

        Shape of time_limit_dones:
        [num_env, num_times], dtype = bool

    Returns:
        adv: advantages, shape: [num_env, num_times]
        ret: returns, shape: [num_env, num_times]

    Cases of done and time_limit_done:
    +-----------------+---------------------------+--------------------+
    |                 | Done=True                 | Done=False         |
    +=================+===========================+====================+
    | Time_done=True  | Done due to time limit    | ------------------ |
    +-----------------+---------------------------+--------------------+
    | Time_done=False | Done due to other reasons | intermediate steps |
    +-----------------+---------------------------+--------------------+
    """
    returns = torch.zeros_like(values)
    not_dones = torch.logical_not(dones)
    not_time_limit_dones = torch.logical_not(time_limit_dones)
    num_times = rewards.shape[1]
    discount = discount_factor * not_dones
    if use_gae:
        gae = 0
        for step in reversed(range(num_times)):
            # the last discount is 0, to ensure the last value is not used
            td = rewards[..., step] + discount[..., step] * values[
                ..., step + 1] - values[..., step]
            gae = td + discount[..., step] * gae_scaling * gae
            gae = gae * not_time_limit_dones[..., step]
            returns[..., step] = gae + values[..., step]
    else:
        returns[..., -1] = values[..., -1]
        # Loop from last
        # Return = Value if current step is the done step
        # Return = reward + discounted next return otherwise
        for step in reversed(range(num_times)):
            returns[..., step] = \
                not_time_limit_dones[..., step] * \
                (rewards[..., step] + discount[..., step] *
                 returns[..., step + 1]) + time_limit_dones[..., step] \
                * values[..., step]

    returns = returns[..., :-1]
    advantages = returns - values[..., :-1]

    return advantages.clone().detach(), returns.clone().detach()


def compute_mc_return(rewards, gamma=1):
    """Compute the true return of a sequence of rewards."""
    returns = torch.zeros_like(rewards)
    num_times = rewards.shape[-1]
    returns[..., -1] = rewards[..., -1]
    for t in reversed(range(num_times - 1)):
        returns[..., t] = rewards[..., t] + gamma * returns[..., t + 1]
    return returns
