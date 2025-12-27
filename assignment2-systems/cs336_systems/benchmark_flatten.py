import os
import torch
import pandas as pd
import torch.multiprocessing as mp
import torch.distributed as dist
from timeit import default_timer
from ddp_naive import setup
from benchmark import get_config, get_batch
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.model import BasicsTransformerLM


def benchmark_naive(
    rank, world_size, backend,
    model_params,
    criterion, x, y,
    results: mp.Queue,
    warmup_steps=5, timesteps=10,
):
    setup(rank, world_size, backend)

    torch.manual_seed(rank)
    device = 'cuda' if backend == 'nccl' else 'cpu'

    model = BasicsTransformerLM(**model_params).to(device)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    optim = AdamW(model.parameters())

    b = x.shape[0]
    chunk_size = b // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank != world_size - 1 else b

    inputs = x[start_idx:end_idx].to(device)
    targets = y[start_idx:end_idx].to(device).long()

    # Warm-up
    for _ in range(warmup_steps):
        model.train()
        optim.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # NOTE: naive version: use all_reduce for each param grad
        #       flattened version: flatten all grads, all_reduce once
        # TODO: A huge memory usage happens here
        flattened_grads = torch._utils._flatten_dense_tensors(
            [param.grad for param in model.parameters()]
        )
        dist.all_reduce(flattened_grads, op=dist.ReduceOp.AVG, async_op=False)
        synced_grads = torch._utils._unflatten_dense_tensors(
            flattened_grads, [param.grad for param in model.parameters()]
        )
        for param, synced_grad in zip(model.parameters(), synced_grads):
            param.grad = synced_grad
        optim.step()
        if backend == 'nccl':
            torch.cuda.synchronize()

    # timed runs
    train_times = 0.
    sync_times = 0.
    for _ in range(timesteps):
        start = default_timer()
        model.train()
        optim.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        sync_start = default_timer()
        # TODO: A huge memory usage happens here
        flattened_grads = torch._utils._flatten_dense_tensors(
            [param.grad for param in model.parameters()]
        )
        dist.all_reduce(flattened_grads, op=dist.ReduceOp.AVG, async_op=False)
        synced_grads = torch._utils._unflatten_dense_tensors(
            flattened_grads, [param.grad for param in model.parameters()]
        )
        for param, synced_grad in zip(model.parameters(), synced_grads):
            param.grad = synced_grad
        sync_end = default_timer()
        sync_times += (sync_end - sync_start)
        optim.step()
        if backend == 'nccl':
            torch.cuda.synchronize()
        end = default_timer()
        train_times += (end - start)
    train_times /= timesteps
    sync_times /= timesteps

    # Gather per-rank (train_time, sync_time) tuples
    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, (train_times, sync_times))

    if rank == 0:
        trains, syncs = zip(*gathered_results)
        avg_train = sum(trains) / world_size * 1000.0  # ms
        avg_sync = sum(syncs) / world_size * 1000.0  # ms
        results.put((round(avg_train, 2), round(avg_sync, 2)))

    dist.barrier()
    dist.destroy_process_group()


def run_benchmark_naive(output_file=f"../result/ddp/flatten_results.md"):
    model_configs, vocab_size, context_length, batch_size, rope_theta = get_config()
    world_size = 2
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'

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
        x= get_batch(batch_size, context_length, vocab_size, device='cpu')
        y = x.clone()
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()

        mp.spawn(
            fn=benchmark_naive,
            args=(world_size, backend, model_params, cross_entropy, x, y, result_queue),
            nprocs=world_size,
            join=True,
        )

        avg_train, avg_sync = result_queue.get()
        results.append({
            'model_size': config['size'],
            'avg_train_time_ms': avg_train,
            'avg_sync_time_ms': avg_sync,
        })
        print(f'  Average training time: {avg_train} ms')
        print(f'  Average sync time: {avg_sync} ms')

    df = pd.DataFrame(results)
    with open(output_file, 'w') as f:
        f.write(df.to_markdown(index=False))

