import torch
import numpy as np
from typing import Dict
from algorithms.cw_offline import util


class StepReplayBuffer:
    def __init__(self, data_info: Dict[str, tuple],
                 data_norm_info: Dict[str, bool] = None,
                 buffer_size: int = 1_000_000,
                 use_priority: bool = False,
                 priority_label: str = "rewards",
                 priority_scaling: float = 1.0,
                 priority_norm: bool = False,
                 normalize_reward: bool = False,
                 dtype="float32", device="cuda"):

        self.buffer_size = buffer_size
        self.dtype, self.device = util.parse_dtype_device(dtype, device)
        self._ptr = 0
        self._size = 0

        self.data_info = data_info
        self.data_norm_info = data_norm_info or {}
        self.has_normalizer = bool(data_norm_info)

        self.replay_buffer = {}
        self.normalizer = {}

        for name, shape in data_info.items():
            if "done" in name or "dones" in name or "terminals" in name:  # for D4RL dataset
                self.replay_buffer[name] = torch.zeros((buffer_size, *shape), dtype=torch.bool, device=self.device)
            elif "idx" in name or "index" in name:
                self.replay_buffer[name] = torch.zeros((buffer_size, *shape), dtype=torch.long, device=self.device)
            else:
                self.replay_buffer[name] = torch.zeros((buffer_size, *shape), dtype=self.dtype, device=self.device)

        self.has_priority = use_priority
        self.priority_label = priority_label
        self.priority_scaling = priority_scaling
        self.priority_norm = priority_norm

        self.normalize_reward = normalize_reward

        if self.has_priority:
            self.replay_priority = torch.zeros((buffer_size,), dtype=torch.float32, device=self.device)
        else:
            self.replay_priority = None

    @torch.no_grad()
    def add(self, data: Dict[str, torch.Tensor]):
        batch_size = len(next(iter(data.values())))
        end = self._ptr + batch_size

        if end <= self.buffer_size:
            for k in self.replay_buffer:
                self.replay_buffer[k][self._ptr:end] = data[k]
        else:
            first_part = self.buffer_size - self._ptr
            second_part = batch_size - first_part
            for k in self.replay_buffer:
                self.replay_buffer[k][self._ptr:] = data[k][:first_part]
                self.replay_buffer[k][:second_part] = data[k][first_part:]

        self._ptr = (self._ptr + batch_size) % self.buffer_size
        self._size = min(self._size + batch_size, self.buffer_size)

        if self.has_priority:
            self.update_priority()
        if self.has_normalizer:
            self.update_buffer_normalizer()

    @torch.no_grad()
    def update_priority(self):
        data_for_priority = self.replay_buffer[self.priority_label][:self._size].float()
        data_for_priority *= self.priority_scaling
        if self.priority_norm:
            data_for_priority = (data_for_priority - data_for_priority.mean()) / (data_for_priority.std() + 1e-8)
        self.replay_priority[:self._size] = torch.nn.functional.softmax(data_for_priority.double(), dim=0)

    @torch.no_grad()
    def update_buffer_normalizer(self):
        for k in self.data_info:
            if self.data_norm_info.get(k, False):
                data = self.replay_buffer[k][:self._size]
                mean = data.mean(dim=0)
                std = data.std(dim=0)
                self.normalizer[k] = (mean, std)

    def normalize_data(self, name, data):
        if self.data_norm_info.get(name, False):
            mean, std = self.normalizer[name]
            return (data - mean) / (std + 1e-6)
        return data

    @torch.no_grad()
    def sample(self, batch_size: int, normalize=False, use_priority=False):
        assert batch_size <= self._size, "Batch size exceeds buffer size"
        if use_priority and self.has_priority:
            idx = self.replay_priority[:self._size].multinomial(batch_size, replacement=False)
        else:
            idx = torch.randint(0, self._size, (batch_size,), device=self.device)

        batch = {}
        for k, buf in self.replay_buffer.items():
            data = buf[idx]
            if normalize:
                data = self.normalize_data(k, data)
            batch[k] = data
        return batch

    def __len__(self):
        return self._size

    def is_full(self):
        return self._size == self.buffer_size

    def clear(self):
        for buf in self.replay_buffer.values():
            buf.zero_()
        self._ptr = 0
        self._size = 0

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")

        reference_key = next(iter(self.data_info))
        n = data[reference_key].shape[0]

        if n > self.buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset")

        torch_data = {}
        for name, shape in self.data_info.items():
            np_array = data[name]

            # Automatically reshape scalar values
            if len(shape) == 1 and np_array.ndim == 1:
                np_array = np_array[..., None]

            # Set dtype dynamically based on content
            if "done" in name or "dones" in name or "terminals" in name:
                dtype = torch.bool
            elif "idx" in name or "index" in name:
                dtype = torch.long
            else:
                dtype = torch.float32
            torch_data[name] = torch.tensor(np_array, dtype=dtype, device=self.device)

        self.add(torch_data)
        print(f"Dataset size: {n}")
        # if self._size != 0:
        #     raise ValueError("Trying to load data into non-empty replay buffer")
        # n = data["observations"].shape[0]
        # if n > self.buffer_size:
        #     raise ValueError("Replay buffer is smaller than the dataset")
        #
        # torch_data = {
        #     "observations": torch.tensor(data["observations"], dtype=torch.float32, device=self.device),
        #     "actions": torch.tensor(data["actions"], dtype=torch.float32, device=self.device),
        #     "rewards": torch.tensor(data["rewards"][..., None], dtype=torch.float32, device=self.device),
        #     "next_observations": torch.tensor(data["next_observations"], dtype=torch.float32, device=self.device),
        #     "terminals": torch.tensor(data["terminals"][..., None], dtype=torch.float32, device=self.device),
        # }
        #
        # self.add(torch_data)
        # print(f"Dataset size: {n}")
