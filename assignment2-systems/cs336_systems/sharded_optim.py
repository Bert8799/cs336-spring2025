import os
import torch
import pandas as pd
import torch.distributed as dist
import torch.multiprocessing as mp
from ddp_naive import setup
from benchmark import get_config, get_batch
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.model import BasicsTransformerLM


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.all_params = list(params)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.param_shards = []
        for i, param in enumerate(self.all_params):
            if i % self.world_size == self.rank:
                self.param_shards.append(param)

        self.optimizer = optimizer_cls([{'params': self.param_shards}], **kwargs)
        self.handles = []

        super(ShardedOptimizer, self).__init__(self.all_params, {})
    
    def step(self, closure=None, **kwargs):
        self.optimizer.step(closure, **kwargs)
        with torch.no_grad():
            for i, param in enumerate(self.all_params):
                src_rank = i % self.world_size
                handle = dist.broadcast(param, src=src_rank, async_op=True)
                self.handles.append(handle)

        for handle in self.handles:
            handle.wait()
        self.handles.clear()
    
    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        # TODO: Handle new param groups in sharded optimizer


def benchmark_sharded_optim(
    rank, world_size, backend,
    model_params, sharded,
    criterion, x, y,
    results: mp.Queue,
    warmup_steps=5, timesteps=10,
):
    setup(rank, world_size, backend)

    torch.manual_seed(rank)
    device = 'cuda'

    model = BasicsTransformerLM(**model_params).to(device)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    if sharded:
        optim = ShardedOptimizer(
            model.parameters(),
            AdamW,
        )
    else:
        optim = AdamW(model.parameters())
    
    torch.cuda.reset_peak_memory_stats()
    init_mem = torch.cuda.memory_allocated() / (1024 ** 3)
    
    b = x.shape[0]
    chunk_size = b // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank != world_size - 1 else b

    inputs = x[start_idx:end_idx].to(device)
    targets = y[start_idx:end_idx].to(device).long()

    def train_step():
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()

    model.train()
    # Warm-up
    for _ in range(warmup_steps):
        optim.zero_grad()
        train_step()
        optim.step()

    torch.cuda.synchronize()
    
    # timed runs
    prev_mem = 0
    post_mem = 0
    for _ in range(timesteps):
        torch.cuda.reset_peak_memory_stats()
        optim.zero_grad()
        train_step()
        prev_mem = max(prev_mem, torch.cuda.max_memory_allocated() / (1024 ** 3))
        torch.cuda.reset_peak_memory_stats()
        optim.step()
        post_mem = max(post_mem, torch.cuda.max_memory_allocated() / (1024 ** 3))
        torch.cuda.synchronize()

    gathered_mem = torch.tensor([init_mem, prev_mem, post_mem], device=device)
    dist.reduce(gathered_mem, dst=0, op=dist.ReduceOp.MAX)
    
    if rank == 0:
        init_mem, prev_mem, post_mem = gathered_mem.tolist()
        results.put({
            'init_mem': init_mem,
            'peak_mem_before_optim': prev_mem,
            'peak_mem_after_optim': post_mem,
        })

    dist.barrier()
    dist.destroy_process_group()


def run_benchmark(output_file=f"../result/ddp/sharded_results.md"):
    model_configs, vocab_size, context_length, batch_size, rope_theta = get_config()
    world_size = 2
    assert torch.cuda.is_available(), "Sharded optimizer requires CUDA backend"
    backend = 'nccl'

    x= get_batch(batch_size, context_length, vocab_size, device='cpu')
    y = x.clone()

    results = []
    for config in model_configs[:-1]:
        print(f"Computing model size: {config['size']}")
        model_params = {
            'vocab_size': vocab_size,
            'context_length': context_length,
            'd_model': config['d_model'],
            'num_layers': config['num_layers'],
            'num_heads': config['num_heads'],
            'd_ff': config['d_ff'],
            'rope_theta': rope_theta,
        }

        for sharded in [True, False]:
            mode = 'sharded' if sharded else 'regular'
            print(f"  Using {mode} optimizer")
            ctx = mp.get_context('spawn')
            result_queue = ctx.Queue()
            mp.spawn(
                fn=benchmark_sharded_optim,
                args=(
                    world_size, backend, 
                    model_params, sharded,
                    cross_entropy, x, y, 
                    result_queue
                ),
                nprocs=world_size,
                join=True,
            )
            result = result_queue.get()
            results.append({
                'model_size': config['size'],
                'optimizer': mode,
                **result,
            })
            print(f"    Init Mem: {result['init_mem']:.4f} GB")
            print(f"    Before Optim: {result['peak_mem_before_optim']:.4f} GB")
            print(f"    After Optim: {result['peak_mem_after_optim']:.4f} GB")
    
    df = pd.DataFrame(results)
    with open(output_file, 'w') as f:
        f.write(df.to_markdown(index=False))


if __name__ == "__main__":
    run_benchmark()