import torch
import timeit
import pandas as pd
from einops import rearrange, einsum, reduce


def get_config():
    # Model configurations
    batch_size = 8
    d_models = [6, 32, 64, 128]
    seq_lens = [64, 128, 256, 512, 1024]

    return batch_size, d_models, seq_lens


assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a GPU."
device = torch.device("cuda")


def softmax(x):
    x_max = reduce(x, 'b seq_q seq_k -> b seq_q 1', 'max')
    x = torch.exp(x - x_max)
    x_sum = reduce(x, 'b seq_q seq_k -> b seq_q 1', 'sum')
    return x / x_sum


def scaled_dot_product_attention(q, k, v):
    d_k = q.size(-1)
    scores = einsum(q, k, "b h seq_q d, b h seq_k d -> b h seq_q seq_k") / (d_k ** 0.5)
    attn_weights = softmax(scores)
    output = einsum(attn_weights, v, "b h seq_q seq_k, b h seq_k d -> b h seq_q d")
    return output


def benchmark_attention(batch_size, d_model, seq_len, warmup_steps=5, timed_steps=10):
    num_heads = 8
    d_head = d_model // num_heads

    q = torch.randn(batch_size, num_heads, seq_len, d_head, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, d_head, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, d_head, device=device)

    # forward
    # Warm-up
    for _ in range(warmup_steps):
        _ = scaled_dot_product_attention(q, k, v)
    
    torch.cuda.synchronize()

    # timed runs
    forward_times = []
    for _ in range(timed_steps):
        start_time = timeit.default_timer()
        _ = scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        forward_times.append(timeit.default_timer() - start_time)
    
    torch.cuda.reset_peak_memory_stats()
    _ = scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    memory_before = torch.cuda.max_memory_allocated() / (1024 ** 3)

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    # backward
    # Warm-up
    for _ in range(warmup_steps):
        output = scaled_dot_product_attention(q, k, v)
        output.mean().backward()

    torch.cuda.synchronize()

    # timed runs
    backward_times = []
    for _ in range(timed_steps):
        start_time = timeit.default_timer()
        output = scaled_dot_product_attention(q, k, v)
        output.mean().backward()
        torch.cuda.synchronize()
        backward_times.append(timeit.default_timer() - start_time)
    
    torch.cuda.reset_peak_memory_stats()
    output = scaled_dot_product_attention(q, k, v)
    output.mean().backward()
    torch.cuda.synchronize()
    memory_after = torch.cuda.max_memory_allocated() / (1024 ** 3)

    avg_forward = (sum(forward_times) / timed_steps) * 1000  # in milliseconds
    avg_backward = (sum(backward_times) / timed_steps) * 1000  # in milliseconds
    
    avg_forward = round(avg_forward, 2)
    avg_backward = round(avg_backward, 2)
    memory_before = round(memory_before, 2)
    memory_after = round(memory_after, 2)

    return {
        "forward_time": avg_forward,
        "backward_time": avg_backward,
        "memory_before": memory_before,
        "memory_after": memory_after,
    }


def run_benchmarks(output_file=f"../result/benchmark/attn_results.md"):
    batch_size, d_models, seq_lens = get_config()
    print(f"Fixed batch size: {batch_size}")

    results = []
    for d_model in d_models:
        for seq_len in seq_lens:
            result = {
                "d_model": d_model,
                "seq_len": seq_len,
            }
            result.update(benchmark_attention(batch_size, d_model, seq_len))
            results.append(result)
            print(f"Completed: d_model={d_model}, seq_len={seq_len}")
            print(f"  Forward Time: {result['forward_time']} ms")
            print(f"  Backward Time: {result['backward_time']} ms")
            print(f"  Memory Before Backward: {result['memory_before']} GB")
            print(f"  Memory After Backward: {result['memory_after']} GB")

    df = pd.DataFrame(results)
    with open(output_file, "w") as f:
        f.write(df.to_markdown(index=False))


if __name__ == "__main__":
    run_benchmarks()
    