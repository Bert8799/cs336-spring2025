import torch
import timeit
import pandas as pd
from contextlib import nullcontext
from statistics import mean, stdev
from torch.amp import autocast, GradScaler
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.model import BasicsTransformerLM


def get_config():
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

    return model_configs, vocab_size, context_length, batch_size, rope_theta


assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a GPU."
device = torch.device("cuda")

mixed = True
target = "mixed" if mixed else "benchmark"
ctx = torch.amp.autocast(device.type, torch.bfloat16) if mixed else nullcontext()
scaler = GradScaler()


def get_batch(batch_size, context_length, vocab_size, device):
    token_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return token_ids


def benchmark_model(model, x, y, mod="forward", warmup_steps=5, timed_steps=10):
    optim = AdamW(model.parameters())
    criterion = cross_entropy
    model.train()
    for _ in range(warmup_steps):
        if mod == "forward":
            with torch.inference_mode():
                with ctx:
                    _ = model(x)
        else:
            optim.zero_grad()
            with ctx:
                outputs = model(x)
                loss = criterion(outputs.view(-1, outputs.size(-1)), y.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
    
    torch.cuda.synchronize(device)
    
    times, memories = [], []
    for _ in range(timed_steps):
        torch.cuda.reset_peak_memory_stats(device)

        start_time = timeit.default_timer()

        if mod == "forward":
            with torch.inference_mode():
                with ctx:
                    _ = model(x)
        else:
            optim.zero_grad()
            with ctx:
                outputs = model(x)
                loss = criterion(outputs.view(-1, outputs.size(-1)), y.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

        torch.cuda.synchronize(device)
        end_time = timeit.default_timer()
        elapsed_time = (end_time - start_time) * 1000
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3) 
        times.append(elapsed_time)
        memories.append(peak_memory)

    return {
        f"{mod}_time_mean": mean(times),
        f"{mod}_time_std": stdev(times),
        f"{mod}_memory": max(memories),
    }
    


def run_benchmark(output_file=f"../result/benchmark/{target}_results.md"):
    model_configs, vocab_size, context_length, batch_size, rope_theta = get_config()
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

        outputs1 = benchmark_model(model, x, y, mod="forward")
        print(f"  Forward Time: {outputs1['forward_time_mean']:.2f} ms ± {outputs1['forward_time_std']:.2f} ms")
        print(f"  Forward Memory: {outputs1['forward_memory']:.2f} GB")
        
        if config["size"] == "2.7B":
            print("  Skipping backward benchmark for 2.7B model due to resource constraints.")
            results.append({
                "size": config["size"],
                "fwd_avg": outputs1["forward_time_mean"],
                "fwd_std": outputs1["forward_time_std"],
                "fwd_mem": outputs1["forward_memory"],
                "bwd_avg": None,
                "bwd_std": None,
                "bwd_mem": None,
            })
            continue

        outputs2 = benchmark_model(model, x, y, mod="backward")
        print(f"  Backward Time: {outputs2['backward_time_mean']:.2f} ms ± {outputs2['backward_time_std']:.2f} ms")
        print(f"  Backward Memory: {outputs2['backward_memory']:.2f} GB")

        results.append({
            "size": config["size"],
            "fwd_avg": outputs1["forward_time_mean"],
            "fwd_std": outputs1["forward_time_std"],
            "fwd_mem": outputs1["forward_memory"],
            "bwd_avg": outputs2["backward_time_mean"],
            "bwd_std": outputs2["backward_time_std"],
            "bwd_mem": outputs2["backward_memory"],
        })

    
    df = pd.DataFrame(results)
    with open(output_file, "w") as f:
        f.write(df.to_markdown(index=False))


if __name__ == "__main__":
    run_benchmark()