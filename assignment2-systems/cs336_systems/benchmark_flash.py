import torch
import triton
import pandas as pd

from cs336_systems.flash_attn import FlashAttentionPyTorch, FlashAttentionTriton


torch.set_float32_matmul_precision('high')


def get_config():
    # Model configurations
    batch_size = 1
    seq_lens = [128, 256, 512, 1024, 4096, 8192]
    dims = [16, 32, 64, 128]
    dtypes = [torch.bfloat16, torch.float32]

    return batch_size, seq_lens, dims, dtypes


assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a GPU."
device = torch.device("cuda")


def benchmark_flash_attention(attn, batch_size, seq_len, dim, dtype):
    Q = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=True)
    dO = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)

    # forward
    fwd_time = triton.testing.do_bench(lambda: attn(Q, K, V, True), warmup=5, rep=10) * 1000
    # backward
    full_time = triton.testing.do_bench(lambda: attn(Q, K, V, True).backward(dO), warmup=5, rep=10) * 1000
    bwd_time = full_time - fwd_time

    fwd_time = round(fwd_time, 2)
    bwd_time = round(bwd_time, 2)

    torch.cuda.empty_cache()

    return {
        "fwd_time": fwd_time,
        "bwd_time": bwd_time
    }


def run_benchmarks(output_file=f"../result/benchmark/attn_triton.md"):
    batch_size, seq_lens, dims, dtypes = get_config()
    print(f'Fixed batch size:{batch_size}')

    attn_pytorch = torch.compile(FlashAttentionPyTorch.apply)
    attn_triton = FlashAttentionTriton.apply

    results = []
    for dtype in dtypes:
        for seq_len in seq_lens:
            for dim in dims:
                print(f'Computing dtype={dtype}, seq_len={seq_len}, dim={dim}')
                if seq_len == 128:
                    # PyTorch implementation, to slow to run for large seq_len
                    res = benchmark_flash_attention(
                        attn_pytorch,
                        batch_size,
                        seq_len,
                        dim,
                        dtype
                    )
                    print(f'  PyTorch - forward: {res["fwd_time"]} ms, backward: {res["bwd_time"]} ms')
                # Triton implementation
                res = {
                    "seq_len": seq_len,
                    "dim": dim,
                    "dtype": str(dtype).split('.')[-1],
                }
                res.update(benchmark_flash_attention(
                    attn_triton,
                    batch_size,
                    seq_len,
                    dim,
                    dtype
                ))
                results.append(res)
                print(f'  Triton  - forward: {res["fwd_time"]} ms, backward: {res["bwd_time"]} ms')
    
    df = pd.DataFrame(results)
    with open(output_file, "w") as f:
        f.write(df.to_markdown(index=False))

if __name__ == "__main__":
    run_benchmarks()