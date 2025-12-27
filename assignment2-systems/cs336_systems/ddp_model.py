import os
import torch
import torch.nn as nn
import torch.distributed as dist


class DDPOverlap(nn.Module):
    def __init__(self, module: nn.Module):
        super(DDPOverlap, self).__init__()
        self.module = module
        self.handles = []

        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)
        
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(self.sync_gradients)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def sync_gradients(self, param):
        with torch.no_grad():
            param.grad /= dist.get_world_size()
        # Use SUM to adapt gloo backend in test
        # TODO: A huge memory usage happens here
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append(handle)
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()


class DDPBucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float = 25):
        super().__init__()
        self.module = module
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.buckets_params = [] # [[param1, param2], [param3, param4], ...]
        self.bucket_buffers = []
        self.handles = []
        self.bucket_ready_count = None
        
        self._sync_model()
        self._partition_buckets()

    def _sync_model(self):
        state_dict = self.module.state_dict()
        for _, tensor in state_dict.items():
            dist.broadcast(tensor, src=0)
        self.module.load_state_dict(state_dict)

    def _partition_buckets(self):
        params = [p for p in self.module.parameters() if p.requires_grad][::-1]
        current_bucket = []
        current_size = 0
        for p in params:
            current_bucket.append(p)
            current_size += p.numel() * p.element_size()
            if current_size >= self.bucket_size_bytes:
                self._create_bucket(current_bucket)
                current_bucket = []
                current_size = 0
        if current_bucket:
            self._create_bucket(current_bucket)

        self.bucket_ready_count = [0] * len(self.buckets_params)

    def _create_bucket(self, params_list):
        self.buckets_params.append(params_list)
        
        total_elements = sum(p.numel() for p in params_list)
        # assume all params are on the same device and dtype
        # TODO: different dtype handling
        buffer = torch.zeros(total_elements, device=params_list[0].device, dtype=params_list[0].dtype)
        self.bucket_buffers.append(buffer)
        
        bucket_idx = len(self.buckets_params) - 1
        for p in params_list:
            p.register_post_accumulate_grad_hook(
                lambda param, b_idx=bucket_idx: self._hook(b_idx)
            )

    def _hook(self, bucket_idx):
        self.bucket_ready_count[bucket_idx] += 1
        if self.bucket_ready_count[bucket_idx] == len(self.buckets_params[bucket_idx]):
            self._sync_bucket(bucket_idx)

    def _sync_bucket(self, bucket_idx):
        params = self.buckets_params[bucket_idx]
        buffer = self.bucket_buffers[bucket_idx]
        
        offset = 0
        with torch.no_grad():
            for p in params:
                numel = p.numel()
                buffer[offset:offset + numel].copy_(p.grad.view(-1))
                offset += numel
            
            buffer.div_(self.world_size)
        
        handle = dist.all_reduce(buffer, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append((handle, bucket_idx))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle, bucket_idx in self.handles:
            handle.wait()
            
            params = self.buckets_params[bucket_idx]
            buffer = self.bucket_buffers[bucket_idx]
            offset = 0
            with torch.no_grad():
                for p in params:
                    numel = p.numel()
                    p.grad.copy_(buffer[offset:offset + numel].view_as(p.grad))
                    offset += numel
        
        self.handles.clear()
        self.bucket_ready_count = [0] * len(self.buckets_params)
            