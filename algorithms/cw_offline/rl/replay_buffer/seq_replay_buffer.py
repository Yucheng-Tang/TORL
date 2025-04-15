import torch

from algorithms.cw_offline import util


class SeqReplayBuffer:
    def __init__(self, data_info: dict,
                 data_norm_info: dict = None,
                 buffer_size: int = 1e4,
                 use_priority: bool = False,
                 priority_label: str = "segment_reward",
                 priority_scaling: float = 1.0,
                 priority_norm: bool = False,
                 dtype="float32", device='cuda',
                 **kwargs):
        self.buffer_size: int = buffer_size
        self.dtype, self.device = util.parse_dtype_device(dtype, device)
        self._ptr = 0
        self._size = 0

        if data_norm_info is not None:
            assert data_info.keys() == data_norm_info.keys(), \
                "data_info and data_norm_info keys must match"
        self.data_info = data_info
        self.data_norm_info = data_norm_info
        self.has_normalizer = data_norm_info is not None
        self.replay_buffer = dict()
        self.normalizer = dict()

        self.random_data_name = None

        for data_name, data_shape in data_info.items():
            if "done" in data_name:
                buffer = torch.zeros((buffer_size, *data_shape),
                                     dtype=torch.bool, device=self.device)
            elif "idx" in data_name or "index" in data_name:
                buffer = torch.zeros((buffer_size, *data_shape),
                                     dtype=torch.long, device=self.device)
            else:
                buffer = torch.zeros((buffer_size, *data_shape),
                                     dtype=self.dtype, device=self.device)
            self.replay_buffer[data_name] = buffer
            self.random_data_name = data_name

        self.has_priority = use_priority

        if self.has_priority:
            assert priority_label in self.replay_buffer.keys(), \
                "Priority label not in data_info"
            self.priority_label = priority_label
            self.priority_scaling = float(priority_scaling)
            self.priority_norm = priority_norm
            self.replay_priority = torch.zeros((buffer_size,),
                                               dtype=torch.float64,
                                               device=self.device)
        else:
            self.replay_priority = None

        policy_scope_factor = kwargs.get("policy_scope_factor", 1)
        assert 0 < policy_scope_factor <= 1, \
            "Policy recent factor must be in (0, 1]"
        if policy_scope_factor < 1:
            assert not self.has_priority, \
                "Policy recent factor not supported with priority"
        self.policy_scope = int(self.buffer_size * policy_scope_factor)

    @torch.no_grad()
    def add(self, dataset_dict: dict):

        num_smp = len(dataset_dict[self.random_data_name])

        if self._ptr + num_smp < self.buffer_size:
            # Do not reset pointer
            for data_name, data_buffer in self.replay_buffer.items():
                try:
                    data_buffer[self._ptr: self._ptr + num_smp] \
                        = dataset_dict[data_name]
                except RuntimeError as e:
                    e.add_note(
                        f"Error in adding !!{data_name}!! to replay buffer: {e}")
                    raise e
        else:
            # Reset pointer
            num_buffer_remain = self.buffer_size - self._ptr
            for data_name, data_buffer in self.replay_buffer.items():
                try:
                    data_buffer[self._ptr:] \
                        = dataset_dict[data_name][:num_buffer_remain]
                    data_buffer[:num_smp - num_buffer_remain] \
                        = dataset_dict[data_name][num_buffer_remain:]
                except RuntimeError as e:
                    e.add_note(
                        f"Error in adding !!{data_name}!! to replay buffer: {e}")
                    raise e
        # Update pointer and size
        self._ptr = (self._ptr + num_smp) % self.buffer_size
        self._size = min(self._size + num_smp, self.buffer_size)

        # Update priority
        if self.has_priority:
            self.update_priority()

        # Update normalizer
        if self.data_norm_info and any(self.data_norm_info.values()):
            self.update_buffer_normalizer()

    def update_buffer_normalizer(self):
        for data_name, data_shape in self.data_info.items():
            if self.data_norm_info[data_name] is True:
                assert len(data_shape) >= 1  # data must have a time dimension
                data = self.replay_buffer[data_name][:self._size]
                mean = data.mean(dim=[0, 1])  # mean over batch and time
                std = data.std(dim=[0, 1])   # std over batch and time
                self.normalizer[data_name] = (mean, std)
            else:
                self.normalizer[data_name] = (None, None)

    def normalize_data(self, data_name, data):
        if self.data_norm_info[data_name]:
            mean = self.normalizer[data_name][0]
            std = self.normalizer[data_name][1]
            return (data - mean) / (std + 1e-6)
        else:
            return data
    @torch.no_grad()
    def update_priority(self):
        if not self.has_priority:
            return

        # Get the values used for priority computation
        data_for_priority = self.replay_buffer[self.priority_label][:self._size]

        # Scale the priority values
        data_for_priority *= self.priority_scaling

        # Normalize the priority values
        if self.priority_norm:
            data_for_priority = (data_for_priority - data_for_priority.mean()) \
                                / data_for_priority.std()

        # buffer not full
        self.replay_priority[:self._size] = torch.nn.LogSoftmax(dim=0)(
            data_for_priority.double()).exp()

    @torch.no_grad()
    def sample(self, batch_size, normalize=False, use_priority=False,
               policy_recent=False):
        assert not (use_priority and policy_recent), "Cannot use both priority and policy recent"
        assert batch_size <= self._size, "Batch size larger than buffer size"
        if use_priority and self.has_priority:
            return self._sample_with_priority(batch_size, normalize)
        elif use_priority and not self.has_priority:
            raise ValueError("No priority in the replay buffer")
        else:
            return self._sample_without_priority(batch_size, normalize,
                                                 policy_recent)

    @torch.no_grad()
    def _sample_without_priority(self, batch_size, normalize,
                                 limited_policy_history=False):
        if limited_policy_history:
            history = min(self._size, self.policy_scope)
            assert history >= batch_size, "Not enough history for policy recent"

            idx = torch.randint(self._size - history, self._size, (batch_size,),
                                device=self.device)
        else:
            idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        smp_dict = dict()
        for data_name, data_buffer in self.replay_buffer.items():
            smp_data = data_buffer[idx]

            # Norm
            if normalize:
                smp_data = self.normalize_data(data_name, smp_data)

            smp_dict[data_name] = smp_data
        return smp_dict

    @torch.no_grad()
    def _sample_with_priority(self, batch_size, normalize):
        assert not normalize, "Not implemented"
        priority = self.replay_priority[:self._size]
        idx = priority.multinomial(num_samples=batch_size, replacement=False)

        smp_dict = dict()
        for data_name, data_buffer in self.replay_buffer.items():
            smp_data = data_buffer[idx]

            # Norm
            if normalize:
                smp_data = self.normalize_data(data_name, smp_data)

            smp_dict[data_name] = smp_data
        return smp_dict

    def __len__(self):
        return self._size

    def is_full(self):
        return self._size == self.buffer_size

    def __getitem__(self, idx):
        return [data_buffer[idx] for data_buffer in self.replay_buffer.values()]

    def current_pointer(self):
        return self._ptr

    @torch.no_grad()
    def clear_buffer(self):
        for data_buffer in self.replay_buffer.values():
            data_buffer.zero_()
        self._ptr = 0
        self._size = 0
