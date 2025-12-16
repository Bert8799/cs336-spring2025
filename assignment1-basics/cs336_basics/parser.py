from __future__ import annotations
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer language model")

    # Model arguments
    parser.add_argument(
        "--vocab_size", type=int, default=10000, help="Size of vocabulary"
    )
    parser.add_argument(
        "--d_model", type=int, default=512, help="Model dimension"
        )
    parser.add_argument(
        "--d_ff", type=int, default=1344, help="FFN dimension"
        )
    parser.add_argument(
        "--context_length", type=int, default=256, help="Maximum sequence length"
    )
    parser.add_argument(
        "--num_heads", type=int, default=16, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of transformer layers"
    )
    parser.add_argument(
        "--rope_theta", type=float, default=10000.0, help="RoPE theta parameter"
    )

    # Optimizer arguments
    parser.add_argument(
        "--max_lr", type=float, default=1e-3, help="Maximum learning rate"
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-4, help="Minimum learning rate"
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=500, help="Warmup iterations"
    )
    parser.add_argument(
        "--cosine_iters", type=int, default=10000, help="Cosine annealing iterations"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="Weight decay"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping norm"
    )

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2, help="Total training steps"
    )
    parser.add_argument(
        "--print_every", type=int, default=100, help="Print every N iterations"
    )

    parser.add_argument(
        "--data_path", type=str, default="./data", help="Path to training data"
    )
    parser.add_argument(
        "--out", type=str, default="./results", help="Output directory"
    )

    return parser.parse_args()