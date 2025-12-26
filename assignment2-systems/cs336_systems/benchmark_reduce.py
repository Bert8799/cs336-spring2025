import os
import torch
import pandas as pd
import multiprocessing as mp
import torch.distributed as dist
from timeit import default_timer


def get_config():
    backends = ['gloo', 'nccl'] if torch.cuda.is_available() else ['gloo']
    tensor_sizes = [1024, 1024 * 1024, 10 * 1024 * 1024]
    world_sizes = 2
    return backends, tensor_sizes, world_sizes


def setup(rank, world_size, backend):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def benchmark_reduce(
    rank, world_size, backend,
    tensor_size,
    results: mp.queues.Queue,
    warmup_steps=5, timesteps=10,
):
    setup(rank, world_size, backend)
    
    device = f'cuda:{rank}' if backend == 'nccl' else 'cpu'
    tensor = torch.randn(tensor_size, device=device)

    # Warm-up
    for _ in range(warmup_steps):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
        if backend == 'nccl':
            torch.cuda.synchronize(device)
    
    # timed runs
    sums = 0.
    for _ in range(timesteps):
        start = default_timer()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
        if backend == 'nccl':
            torch.cuda.synchronize(device)
        end = default_timer()
        sums += (end - start)
    sums /= timesteps

    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, sums)

    if rank == 0:
        avg_time = sum(gathered_results) / world_size * 1000  # convert to milliseconds
        avg_time = round(avg_time, 2)
        results.put(avg_time)


def run_benchmark(output_file=f"../result/ddp/reuduce_results.md"):
    backends, tensor_sizes, world_size = get_config()
    print(f'Fixed world size: {world_size}')
    results = []
    for backend in backends:
        for tensor_size in tensor_sizes:
            print(f'Computing backend={backend}, tensor_size={tensor_size}')
            res = {
                'backend': backend,
                'tensor_size': tensor_size,
            }
            ctx = mp.get_context('spawn')
            result_queue = ctx.Queue()

            mp.spawn(
                fn=benchmark_reduce,
                args=(world_size, backend, tensor_size, result_queue),
                nprocs=world_size,
                join=True,
            )

            avg_time = result_queue.get()
            res.update({'avg_time_ms': avg_time})
            results.append(res)
            print(f'  Average time: {avg_time} ms')

    df = pd.DataFrame(results)
    with open(output_file, 'w') as f:
        f.write(df.to_markdown(index=False))

        
if __name__ == '__main__':
    run_benchmark()
