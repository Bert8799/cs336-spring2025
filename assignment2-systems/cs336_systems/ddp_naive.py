import os
import torch
import pandas as pd
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from copy import deepcopy
from multiprocessing import Manager
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.model import Linear, RMSNorm, SwiGLU


class SimpleModel(nn.Module):
    def __init__(self, in_features, hidden_dims, out_features):
        super(SimpleModel, self).__init__()
        self.fc = Linear(in_features, hidden_dims)
        self.norm = RMSNorm(hidden_dims)
        self.ffn = SwiGLU(hidden_dims, out_features)

    def forward(self, x):
        x = self.fc(x)
        x = self.norm(x)
        x = self.ffn(x)
        return x


def train(model, optim, criterion, x, y, iterations, print_every):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    model.train()

    inputs = x.to(device)
    # Ensure class indices dtype is int64 for cross_entropy
    targets = y.to(device).long()

    for it in range(iterations):
        optim.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optim.step()

        if it % print_every == 0:
            print(f"Iteration {it}, Loss: {loss.item()}")

    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def setup(rank, world_size, backend):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def naive_dpp(
    rank, world_size, backend,
    model,
    criterion, x, y, iterations, print_every,
    results,
):
    setup(rank, world_size, backend)
    
    torch.manual_seed(42 + rank)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = deepcopy(model)
    model.to(device)

    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # Create a fresh optimizer local to this process to avoid
    # cross-device state from a parent process.
    optim = AdamW(model.parameters())

    b, d = x.shape
    chunk_size = b // world_size
    start = rank * chunk_size
    end = start + chunk_size if rank != world_size - 1 else b

    inputs = x[start:end].to(device)
    # Ensure class indices dtype is int64 for cross_entropy
    targets = y[start:end].to(device).long()

    model.train()
    for it in range(iterations):
        optim.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG, async_op=False)

        optim.step()

        if rank == 0 and it % print_every == 0:
                print(f"Iteration {it}, Loss: {loss.item()}")
    
    if rank == 0:
        model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        results.put(model_state)
    
    dist.barrier()
    dist.destroy_process_group()


def run_test():
    in_features = 16
    hidden_dims = 32
    out_features = 8
    batch_size = 64
    iterations = 20
    print_every = 5
    world_size = 2
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'

    x = torch.randn(batch_size, in_features)
    # Targets should be integer class indices in [0, out_features)
    y = torch.randint(low=0, high=out_features, size=(batch_size,))

    criterion = cross_entropy

    # Prepare and share an identical initial state across runs
    init_model = SimpleModel(in_features, hidden_dims, out_features)

    # Run reference training
    model_ref = deepcopy(init_model)
    optim_ref = AdamW(model_ref.parameters())
    ref = train(model_ref, optim_ref, criterion, x, y, iterations, print_every)

    # Run naive DDP
    # Use multiprocessing Manager to share results queue
    # Otherwise, when you put a Tensor into a Queue, 
    # PyTorch does not copy the entire contents of the Tensor, 
    # but only sends a handle pointing to the shared memory.
    # But since mp.spawn(join=True) ensures that 
    # the child processes have completely exited, 
    # the shared memory has already been reclaimed.
    manager = Manager()
    result_queue = manager.Queue()

    mp.spawn(
        fn=naive_dpp,
        args=(
            world_size, backend,
            init_model,
            criterion, x, y, iterations, print_every,
            result_queue,
        ),
        nprocs=world_size,
        join=True,
    )

    ans = result_queue.get()

    for k in ref.keys():
        assert torch.allclose(ref[k], ans[k], atol=1e-6), f"Mismatch in parameter {k}"
    
    print("Naive DDP test passed!")


if __name__ == "__main__":
    run_test()