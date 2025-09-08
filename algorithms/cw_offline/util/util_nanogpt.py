"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F

from algorithms.cw_offline import util
import pickle as pkl
from addict import Dict


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias) # Note: needs special init

        # regularization
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.shape[:-2], x.shape[-2], x.shape[-1]

        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)

        # (*B, T, n_head, C / n_head) -> (*B, n_head, T, C / n_head)
        k = k.view(*B, T, self.n_head, C // self.n_head).transpose(-3, -2)
        q = q.view(*B, T, self.n_head, C // self.n_head).transpose(-3, -2)
        v = v.view(*B, T, self.n_head, C // self.n_head).transpose(-3, -2)

        dropout_p = self.dropout if self.training else 0

        # Causal self-attention, Shape of y (*B, nh, T, d_k)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p,
            is_causal=True)
        # y = torch.nn.functional.scaled_dot_product_attention(
        #     query=x, key=x, value=x, attn_mask=None, dropout_p=dropout_p,
        #     is_causal=True)

        # (*B, n_head, T, d_k) -> (*B, T, n_head, d_k)
        # -> (*B, T, n_head, C)
        y = y.transpose(-3, -2).contiguous().view(*B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        bias = config.bias
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.use_layer_norm = config.use_layer_norm
        if self.use_layer_norm:
            self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        if self.use_layer_norm:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:
            x = x + self.attn(x)
            x = x + self.mlp(x)
        return x


class TrajectoryQfunctionGPT(nn.Module):

    def __init__(self, **config):
        super().__init__()

        config = Dict(config)
        self.config = config
        self.use_layer_norm = config.use_layer_norm

        module_dict = dict(
            state_encoder=nn.Linear(config.state_dim, config.n_embd, bias=False),
            action_encoder=nn.Linear(config.action_dim, config.n_embd, bias=False),
            pos_enc=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        )

        if not self.use_layer_norm:
            del module_dict['ln_f']

        self.transformer = nn.ModuleDict(module_dict)
        self.output_layer = nn.Linear(config.n_embd, 1, bias=False)

        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # device and dtype?
        dtype, device = util.parse_dtype_device(config.dtype, config.device)
        self.dtype, self.device = dtype, device

        self.transformer.to(device=device, dtype=dtype)
        self.output_layer.to(device=device, dtype=dtype)

        self.gpt_name = config.name + "_gpt"

        self.relative_pos = config.relative_pos

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.0) # 0.02
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.0) # 0.02

    def forward(self, d_state, c_state, actions, idx_d, idx_c, idx_a):
        """

        Args:
            d_state: decision state, shape [*add_dim, state_dim]
            c_state: current state, [*add_dim, state_dim]
            actions: action sequences, [*add_dim, num_actions, action_dim]
            idx_d: time step index of decision state, [*add_dim]
            idx_c: time step index of current state, [*add_dim]
            idx_a: time step indices of actions, [*add_dim, num_actions]

        Returns:

        """

        if actions is None:
            assert idx_a is None
            t = 0
        else:
            t = actions.size(-2)

        assert t + 2 <= self.config.block_size

        # data embedding,
        # d_state_emd = self.transformer.state_encoder(d_state).unsqueeze(-2)
        state_emd = self.transformer.state_encoder(c_state).unsqueeze(-2)
        if actions is not None:
            action_emd = self.transformer.action_encoder(actions)
            # Shape [*add_dim, 2 + t, n_embed]
            # seq_emb = torch.cat([d_state_emd, state_emd, action_emd], dim=-2)
            seq_emb = torch.cat([state_emd, action_emd], dim=-2)

            # Shape [*add_dim, 2 + t]
            # seq_pos = torch.cat([idx_d[..., None], idx_c[..., None], idx_a], dim=-1)
            seq_pos = torch.cat([idx_c[..., None], idx_a], dim=-1)

        else:
            seq_emb = state_emd
            seq_pos = idx_c[..., None]

        # Double check the relative pos if the decision state is included
        if self.relative_pos:
            # seq_pos = seq_pos - seq_pos[..., 0:1]

            # shape (1, 1+t)
            seq_pos = torch.arange(0, 1 + t, dtype=torch.long,
                                   device=self.device)[None]

        # Shape [*add_dim, t+2, n_embed]
        seq_pos_emb = self.transformer.pos_enc(seq_pos)

        x = self.transformer.drop(seq_emb + seq_pos_emb)

        for block in self.transformer.h:
            x = block(x)

        if self.use_layer_norm:
            # Shape [*add_dim, t+2, n_embed]
            x = self.transformer.ln_f(x)

        # Shape [*add_dim, t+2, n_embed] -> # Shape [*add_dim, t+2, 1]
        # -> [*add_dim, t+2]
        x = self.output_layer(x).squeeze(-1)  # value is dimensionless

        # v = x[..., 1]  # shape [*add_dim, 1]
        # q = x[..., 2:]  # shape [*add_dim, t]
        # return x[..., 1:]
        return x  # now the decision state is removed

    def configure_optimizer(self, weight_decay, learning_rate, betas, device_type):
        # start with all the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def save(self, log_dir: str, epoch: int):
        """
        Save NN structure and weights to file
        Args:
            log_dir: directory to save weights to
            epoch: training epoch

        Returns:
            None
        """
        # Get paths to structure parameters and weights respectively
        s_path, w_path = util.get_nn_save_paths(log_dir, self.gpt_name, epoch)

        # Store structure parameters
        with open(s_path, "wb") as f:
            parameters = self.config
            pkl.dump(parameters, f)

        # Store NN weights
        with open(w_path, "wb") as f:
            torch.save(self.state_dict(), f)

    def load(self, log_dir: str, epoch: int):
        """
        Load NN structure and weights from file
        Args:
            log_dir: directory stored weights
            epoch: training epoch

        Returns:
            None
        """
        # Get paths to structure parameters and weighs respectively
        s_path, w_path = util.get_nn_save_paths(log_dir, self.gpt_name, epoch)

        # Check structure parameters
        with open(s_path, "rb") as f:
            parameters = pkl.load(f)
            assert self.config == parameters, \
                "NN structure parameters do not match"

        # Load NN weights
        self.load_state_dict(torch.load(w_path))
