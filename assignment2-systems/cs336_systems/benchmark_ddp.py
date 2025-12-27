import os
import torch
import pandas as pd
import torch.multiprocessing as mp
import torch.distributed as dist
from copy import deepcopy
from ddp_naive import setup
from timeit import default_timer
from benchmark import get_config, get_batch
from ddp_model import DDPOverlap, DDPBucketed
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.model import BasicsTransformerLM


def benchmark_overlap(
    rank, world_size, backend,
    model_params,
    criterion, x, y,
    results: mp.Queue,
    warmup_steps=5, timesteps=10,
):
    setup(rank, world_size, backend)

    torch.manual_seed(rank)
    device = 'cuda' if backend == 'nccl' else 'cpu'

    non_flatten_model = BasicsTransformerLM(**model_params).to(device)
    non_overlap_model = deepcopy(non_flatten_model)
    for param in non_overlap_model.parameters():
        dist.broadcast(param.data, src=0)

    ddp_base = deepcopy(non_flatten_model)
    ddp_model = DDPOverlap(ddp_base).to(device)

    b = x.shape[0]
    chunk_size = b // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank != world_size - 1 else b

    inputs = x[start_idx:end_idx].to(device)
    targets = y[start_idx:end_idx].to(device).long()

    non_flatten_optim = AdamW(non_flatten_model.parameters())
    non_overlap_optim = AdamW(non_overlap_model.parameters())
    ddp_optim = AdamW(ddp_model.parameters())

    def non_flatten_step():
        non_flatten_model.train()
        non_flatten_optim.zero_grad()
        outputs = non_flatten_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        for param in non_flatten_model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)
        non_flatten_optim.step()
        if backend == 'nccl':
            torch.cuda.synchronize()

    def non_overlap_step():
        non_overlap_model.train()
        non_overlap_optim.zero_grad()
        outputs = non_overlap_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # Flatten and all-reduce gradients
        flattened_grads = torch._utils._flatten_dense_tensors(
            [param.grad for param in non_overlap_model.parameters()]
        )
        dist.all_reduce(flattened_grads, op=dist.ReduceOp.AVG, async_op=False)
        synced_grads = torch._utils._unflatten_dense_tensors(
            flattened_grads, [param.grad for param in non_overlap_model.parameters()]
        )
        for param, synced_grad in zip(non_overlap_model.parameters(), synced_grads):
            param.grad = synced_grad
        non_overlap_optim.step()
        if backend == 'nccl':
            torch.cuda.synchronize()
    
    def ddp_step():
        ddp_model.train()
        ddp_optim.zero_grad()
        outputs = ddp_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        ddp_optim.step()
        if backend == 'nccl':
            torch.cuda.synchronize()

    # Warm-up
    for _ in range(warmup_steps):
        # Non-parallel step
        non_flatten_step()

        # Non-overlap step
        non_overlap_step()

        # Overlap step
        ddp_step()
    
    # Timed runs
    non_flatten_times = 0.
    non_overlap_times = 0.
    ddp_times = 0.
    for _ in range(timesteps):
        start = default_timer()
        non_flatten_step()
        end = default_timer()
        non_flatten_times += (end - start)

        start = default_timer()
        non_overlap_step()
        end = default_timer()
        non_overlap_times += (end - start)

        start = default_timer()
        ddp_step()
        end = default_timer()
        ddp_times += (end - start)

    non_flatten_times /= timesteps
    non_overlap_times /= timesteps
    ddp_times /= timesteps

    # Gather per-rank (train_time, sync_time) tuples
    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, (non_flatten_times, non_overlap_times, ddp_times))

    if rank == 0:
        non_flatten, non_overlap, ddp = zip(*gathered_results)
        avg_non_flatten = sum(non_flatten) / world_size * 1000.0  # ms
        avg_overlap = sum(non_overlap) / world_size * 1000.0  # ms
        avg_ddp = sum(ddp) / world_size * 1000.0  # ms
        results.put((round(avg_non_flatten, 2), round(avg_overlap, 2), round(avg_ddp, 2)))

    dist.barrier()
    dist.destroy_process_group()


def benchmark_bucketed(
    rank, world_size, backend,
    model_params, bucket_size_mb,
    criterion, x, y,
    results: mp.Queue,
    warmup_steps=5, timesteps=10,
):
    setup(rank, world_size, backend)

    torch.manual_seed(rank)
    device = 'cuda' if backend == 'nccl' else 'cpu'

    ddp_base = BasicsTransformerLM(**model_params).to(device)
    ddp_model = DDPBucketed(ddp_base, bucket_size_mb).to(device)

    b = x.shape[0]
    chunk_size = b // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank != world_size - 1 else b

    inputs = x[start_idx:end_idx].to(device)
    targets = y[start_idx:end_idx].to(device).long()

    ddp_optim = AdamW(ddp_model.parameters())

    def ddp_step():
        ddp_model.train()
        ddp_optim.zero_grad()
        outputs = ddp_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        ddp_optim.step()
        if backend == 'nccl':
            torch.cuda.synchronize()

    # Warm-up
    for _ in range(warmup_steps):
        # Overlap step
        ddp_step()
    
    # Timed runs
    ddp_times = 0.
    for _ in range(timesteps):
        start = default_timer()
        ddp_step()
        end = default_timer()
        ddp_times += (end - start)

    ddp_times /= timesteps

    # Gather per-rank (train_time, sync_time) tuples
    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, ddp_times)

    if rank == 0:
        ddp = gathered_results
        avg_ddp = sum(ddp) / world_size * 1000.0  # ms
        results.put(round(avg_ddp, 2))

    dist.barrier()
    dist.destroy_process_group()


def run_benchmark(output_file=f"../result/ddp/bucketed_results.md", mode='bucketed'):
    model_configs, vocab_size, context_length, batch_size, rope_theta = get_config()
    world_size = 2
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    bucket_size_mbs = [1., 10., 100., 1000.]

    x= get_batch(batch_size, context_length, vocab_size, device='cpu')
    y = x.clone()

    results = []
    for config in model_configs[:-2]:
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
        if mode == 'overlap':
            ctx = mp.get_context('spawn')
            result_queue = ctx.Queue()

            mp.spawn(
                fn=benchmark_overlap,
                args=(world_size, backend, model_params, cross_entropy, x, y, result_queue),
                nprocs=world_size,
                join=True,
            )

            avg_non_flatten, avg_non_overlap, avg_ddp = result_queue.get()
            results.append({
                'model_size': config['size'],
                'avg_non_flatten_time_ms': avg_non_flatten,
                'avg_non_overlap_time_ms': avg_non_overlap,
                'avg_ddp_time_ms': avg_ddp,
            })
            print(f'  Average non-flatten time: {avg_non_flatten} ms')
            print(f'  Average non-overlap time: {avg_non_overlap} ms')
            print(f'  Average ddp time: {avg_ddp} ms')
        else:
            for bucket_size_mb in bucket_size_mbs:
                print(f'  Bucket size: {bucket_size_mb} MB')
                ctx = mp.get_context('spawn')
                result_queue = ctx.Queue()

                mp.spawn(
                    fn=benchmark_bucketed,
                    args=(
                        world_size, backend, 
                        model_params, bucket_size_mb,
                        cross_entropy, x, y, 
                        result_queue
                    ),
                    nprocs=world_size,
                    join=True,
                )

                avg_ddp = result_queue.get()
                results.append({
                    'model_size': config['size'],
                    'bucket_size_mb': bucket_size_mb,
                    'avg_time_ms': avg_ddp,
                })
                print(f'  Average time: {avg_ddp} ms')

    df = pd.DataFrame(results)
    with open(output_file, 'w') as f:
        f.write(df.to_markdown(index=False))


if __name__ == '__main__':
    run_benchmark()