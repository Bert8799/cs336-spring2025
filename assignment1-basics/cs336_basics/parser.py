from __future__ import annotations
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer language model")

    # Model arguments
    parser.add_argument(
        "--vocab_size", type=int, default=10_000, help="Size of vocabulary"
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

    # Data arguments
    parser.add_argument(
        "--data_path", type=str, default="./data", help="Path to training data"
    )

    # Optimizer arguments
    parser.add_argument(
        "--max_lr", type=float, default=5e-4, help="Maximum learning rate"
    )
    parser.add_argument(
        "--min_lr", type=float, default=5e-5, help="Minimum learning rate"
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=2000, help="Warmup iterations"
    )
    parser.add_argument(
        "--cosine_iters", type=int, default=18000, help="Cosine annealing iterations"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Weight decay"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping norm"
    )

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size"
    )
    parser.add_argument(
        "--iterations", type=int, default=20000, help="Total training steps"
    )

    # Validation arguments
    parser.add_argument(
        "--validation", type=bool, default=True, help="Enable or disable validation"
    )
    parser.add_argument(
        "--val_every", type=int, default=2000, help="Validate every N iterations"
    )
    parser.add_argument(
        "--val_iters", type=int, default=200, help="Validate every N iterations"
    )

    # Checkpoint arguments
    parser.add_argument(
        "--save_every", type=int, default=2000, help="Save model every N iterations"
    )
    parser.add_argument(
        "--out_path", type=str, default="./result/transformer", help="Output directory"
    )

    return parser.parse_args()