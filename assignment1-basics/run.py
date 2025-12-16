import torch
import numpy as np
from cs336_basics.train_bpe import train_bpe, hf_train_bpe
from cs336_basics.tokenizer import Tokenizer, HuggingFaceTokenizer
from cs336_basics.solver import Solver
from cs336_basics.transformer import TransformerLM
from cs336_basics.parser import parse_args


def run_train_bpe(
    input_path,
    vocab_size,
    special_tokens,
    merge_outpath=None,
    vocab_outpath=None,
    type: str = "custom" or "hf" or None,
) -> tuple[dict[int, str], list[tuple[bytes, bytes]]]:
    if type is None:
        raise ValueError("Type must be specified as 'custom' or 'hf'.")
    elif type == "custom":
        return train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            merge_outpath=merge_outpath,
            vocab_outpath=vocab_outpath
        )
    elif type == "hf":
        return hf_train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            merge_outpath=merge_outpath,
            vocab_outpath=vocab_outpath
        )


def run_tokenize_bpe(
    input_path: str,
    vocab_filepath: str,
    merges_filepath: str,
    type: str = "custom" or "hf" or None,
):
    token_ids = None
    if type is None:
        raise ValueError("Type must be specified as 'custom' or 'hf'.")
    elif type == "hf":
        tokenizer = HuggingFaceTokenizer.from_files(
            vocab_filepath=vocab_filepath,
            merges_filepath=merges_filepath
        )
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        token_ids = tokenizer.encode(text)
    else:
        tokenizer = Tokenizer.from_files(
            vocab_filepath=vocab_filepath,
            merges_filepath=merges_filepath
        )
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        token_ids = tokenizer.encode(text)
    np.save(f"{input_path}.npy", np.array(token_ids, dtype=np.uint16))


def run_train_lm():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        context_length=args.context_length,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        theta=args.rope_theta,
        device=device
    )

    # Load your training and test data here
    train_data = np.memmap(f'{args.data_path}/train.npy', dtype=np.uint16, mode='r')
    test_data = np.memmap(f'{args.data_path}/test.npy', dtype=np.uint16, mode='r')

    solver = Solver(
        model=model,
        train_data=train_data,
        test_data=test_data,
        batch_size=args.batch_size,
        context_length=args.context_length,
        num_epochs=args.num_epochs,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_iters=args.warmup_iters,
        cosine_iters=args.cosine_iters,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        device=device,
        print_every=args.print_every
    )

    solver.train()


if __name__ == "__main__":
    run_train_lm()
