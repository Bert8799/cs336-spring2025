from cs336_basics.train_bpe import train_bpe


def run_train_bpe(
    input_path,
    vocab_size,
    special_tokens,
    merge_outpath=None,
    vocab_outpath=None
) -> tuple[dict[int, str], list[tuple[bytes, bytes]]]:
    return train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        merge_outpath=merge_outpath,
        vocab_outpath=vocab_outpath
    )


if __name__ == "__main__":
    run_train_bpe(
        input_path='data/TinyStoriesV2-GPT4-valid.txt',
        vocab_size=10_000,
        special_tokens=["<|endoftext|>","<|endoftext|><|endoftext|>"],
        merge_outpath='result/TinyStories-valid_bpe_merges.yaml',
        vocab_outpath='result/TinyStories-valid_bpe_vocab.yaml'
    )