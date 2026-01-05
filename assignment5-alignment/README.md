**Note on Environment Configuration:**

This project requires specific adjustments due to **environment conflicts** and **network constraints**:

1. **Connectivity**: Due to GitHub access issues, `alpaca-eval` is sourced from a **local path** instead of the remote repository.
2. **Installation Failures**: `flash-attn` fails to build locally due to environment mismatch; we use a **precompiled local wheel** instead.
3. **CUDA Symbol Conflict**: The system **CUDA 12.0** library shadows the **CUDA 12.4** runtime required by PyTorch/vLLM, causing an `ImportError`. This is resolved by manually prioritizing the virtual environment’s library paths in `LD_LIBRARY_PATH`.

---

**Setup Steps:**

1. **Download `alpaca-eval`**

```bash
git clone -b forward_kwargs_to_vllm https://github.com/nelson-liu/alpaca_eval.git
```

2. **Download `flash-attn.whl`**
   Find the correct wheel version according to your **CUDA, Torch, and CPython versions** at [flash-attn releases](https://github.com/Dao-AILab/flash-attention/releases).

3. **Modify `pyproject.toml`**

```diff
-alpaca-eval = { git = "https://github.com/nelson-liu/alpaca_eval.git", rev = "forward_kwargs_to_vllm" }

+alpaca-eval = { path = "path/to/alpaca_eval" }
+flash-attn = { path = "path/to/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" }
```

4. **Fix library path before syncing**
   This resolves the `libcusparse` symbol error:

```bash
export VENV_LIB="$PWD/.venv/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$VENV_LIB/nvjitlink/lib:$VENV_LIB/cusparse/lib:$LD_LIBRARY_PATH"
uv sync
```

---

# CS336 Spring 2025 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)

We include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run unit tests:

``` sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

