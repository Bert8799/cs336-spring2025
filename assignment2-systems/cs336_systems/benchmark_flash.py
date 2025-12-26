import torch
import triton
import pandas as pd

from cs336_systems.flash_attn import FlashAttentionPyTorch, FlashAttentionTriton


def get_config():
    # Model configurations
    batch_size = 1
    seq_lens = [128, 256, 512, 1024, 4096, 8192]
    dims = [16, 32, 64, 128]
    dtypes = [torch.bfloat16, torch.float32]

    return batch_size, seq_lens, dims, dtypes


assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a GPU."
device = torch.device("cuda")


def benchmark_flash_attention(impl1, impl2, batch_size, seq_len, dim, dtype):
    Q = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)

    attention1 = torch.compile(impl1)
    attention2 = torch.compile(impl2)
    # forward
    fwd_time1 = triton.testing.do_bench(lambda: attention1(Q, K, V, True), warmup=5, rep=10) * 1000
    fwd_time2 = triton.testing.do_bench(lambda: attention2(Q, K, V, True), warmup=5, rep=10) * 1000
    # backward
    def backward1():
        out = attention1(Q, K, V, True)
        out.sum().backward()

    def backward2():
        out = attention2(Q, K, V, True)
        out.sum().backward()
    
    full_time1 = triton.testing.do_bench(backward1, warmup=5, rep=10) * 1000
    full_time2 = triton.testing.do_bench(backward2, warmup=5, rep=10) * 1000

    bwd_time1 = full_time1 - fwd_time1
    bwd_time2 = full_time2 - fwd_time2

    fwd_time1 = round(fwd_time1, 2)
    fwd_time2 = round(fwd_time2, 2)
    bwd_time1 = round(bwd_time1, 2)
    bwd_time2 = round(bwd_time2, 2)

    torch.cuda.empty_cache()

    return {
        "fwd_time1": fwd_time1,
        "fwd_time2": fwd_time2,
        "bwd_time1": bwd_time1,
        "bwd_time2": bwd_time2
    }


def run_benchmarks(output_file=f"../result/benchmark/flash_attn.md"):
    batch_size, seq_lens, dims, dtypes = get_config()
    print(f'Fixed batch size:{batch_size}')

    results = []
    for dtype in dtypes:
        for seq_len in seq_lens:
            for dim in dims:
                res = {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "dim": dim,
                    "dtype": str(dtype).split('.')[-1],
                }
                res.update(benchmark_flash_attention(
                    FlashAttentionPyTorch.apply,
                    FlashAttentionTriton.apply,
                    batch_size,
                    seq_len,
                    dim,
                    dtype
                ))
                results.append(res)
                print(f'Completed dtype={dtype}, seq_len={seq_len}, dim={dim}')
                print(f'    PyTorch - forward: {res["fwd_time1"]} ms, backward: {res["bwd_time1"]} ms')
                print(f'    Triton  - forward: {res["fwd_time2"]} ms, backward: {res["bwd_time2"]} ms')
    
    df = pd.DataFrame(results)
    with open(output_file, "w") as f:
        f.write(df.to_markdown(index=False))

if __name__ == "__main__":
    run_benchmarks()