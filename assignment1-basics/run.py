from cs336_basics.train_bpe import train_bpe
from cs336_basics.hf_train_bpe import hf_train_bpe


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


if __name__ == "__main__":
    run_train_bpe_dataset = "openwebtext"

    run_train_bpe_type = "hf"
    run_train_bpe_path = f"{run_train_bpe_type}_" if run_train_bpe_type != "custom" else ""

    run_train_bpe(
        input_path="data/owt_train.txt",
        vocab_size=32_000,
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"],
        merge_outpath=f"result/tokenizer/{run_train_bpe_dataset}/{run_train_bpe_path}train_merges.yaml",
        vocab_outpath=f"result/tokenizer/{run_train_bpe_dataset}/{run_train_bpe_path}train_vocab.yaml",
        type=run_train_bpe_type
    )

