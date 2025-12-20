import torch
import timeit
import pandas as pd
from statistics import mean, stdev
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.model import BasicsTransformerLM


# Model configurations
model_configs = [
    {"size": "small" , "d_model": 768 , "d_ff": 3072 , "num_layers": 12, "num_heads": 12},
    {"size": "medium", "d_model": 1024, "d_ff": 4096 , "num_layers": 24, "num_heads": 16},
    {"size": "large" , "d_model": 1280, "d_ff": 5120 , "num_layers": 36, "num_heads": 20},
    {"size": "xl"    , "d_model": 1600, "d_ff": 6400 , "num_layers": 48, "num_heads": 25},
    {"size": "2.7B"  , "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
]

# Hyperparameters
vocab_size = 10_000
context_length = 256
batch_size = 4
rope_theta = 10_000.0

warmup_steps = 5
timed_steps = 10


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_batch(batch_size, context_length, vocab_size, device):
    token_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return token_ids


def benchmark_model(model, x, y, mod="forward"):
    optim = AdamW(model.parameters())
    criterion = cross_entropy

    def forward_pass():
        _ = model(x)

    def backward_pass():
        optim.zero_grad()
        outputs = model(x)
        loss = criterion(outputs.view(-1, outputs.size(-1)), y.view(-1))
        loss.backward()
        optim.step()

    if mod == "forward":
        model.eval()
    elif mod == "backward":
        model.train()
    else:
        raise ValueError("mod must be either 'forward' or 'backward'")
    
    # Warm-up
    for _ in range(warmup_steps):
        if mod == "forward":
            forward_pass()
        elif mod == "backward":
            backward_pass()
    
    # Timed runs
    times, memories = [], []

    if mod == "forward":
        for _ in range(timed_steps):
            torch.cuda.reset_peak_memory_stats(device=device) if device.type == 'cuda' else None
            start = timeit.default_timer()
            with torch.inference_mode():
                forward_pass()
            torch.cuda.synchronize(device) if device.type == 'cuda' else None
            end = timeit.default_timer()
            times.append(end - start)
            memories.append((torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)) if device.type == 'cuda' else 0)
    elif mod == "backward":
        for _ in range(timed_steps):
            torch.cuda.reset_peak_memory_stats(device=device) if device.type == 'cuda' else None
            start = timeit.default_timer()
            backward_pass()
            torch.cuda.synchronize(device) if device.type == 'cuda' else None
            end = timeit.default_timer()
            times.append(end - start)
            memories.append((torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)) if device.type == 'cuda' else 0)

    return mean(times), stdev(times), mean(memories)


def run_benchmark(output_file="../results/benchmark/benchmark_results.md"):
    results = []
    for config in model_configs:
        print(f"Benchmarking {config['size']} model:")
        model = BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            rope_theta=rope_theta
        ).to(device)

        x = get_batch(batch_size, context_length, vocab_size, device)
        y = x.clone()

        fwd_time_mean, fwd_time_std, fwd_memory = benchmark_model(model, x, y, mod="forward")
        print(f"  Forward Pass: {fwd_time_mean:.4f} s ± {fwd_time_std:.4f} s, Memory: {fwd_memory:.4f} GB")

        if config["size"] == "2.7B":
            bwd_time_mean, bwd_time_std, bwd_memory = float('nan'), float('nan'), float('nan')
            print("  Skipping backward pass for 2.7B model due to memory constraints.")
        else:
            bwd_time_mean, bwd_time_std, bwd_memory = benchmark_model(model, x, y, mod="backward")
            print(f"  Backward Pass: {bwd_time_mean:.4f} s ± {bwd_time_std:.4f} s, Memory: {bwd_memory:.4f} GB")
        print()

        results.append({
            "model_size": config["size"],
            "forward_time_mean": fwd_time_mean,
            "forward_time_std": fwd_time_std,
            "forward_memory": fwd_memory,
            "backward_time_mean": bwd_time_mean,
            "backward_time_std": bwd_time_std,
            "backward_memory": bwd_memory,
        })
    
    df = pd.DataFrame(results)
    with open(output_file, "w") as f:
        f.write(df.to_markdown(index=False))


if __name__ == "__main__":
    run_benchmark()