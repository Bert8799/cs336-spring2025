import yaml
import time
import json
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel


def hf_train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    merge_outpath: str | None = None,
    vocab_outpath: str | None = None,
):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens or ["<|endoftext|>"]
    )

    start_time = time.time()
    tokenizer.train(files=[input_path], trainer=trainer)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # get merges and vocab
    model_file = "temp_tokenizer.json"
    tokenizer.save(model_file)
    with open(model_file, "r", encoding="utf-8") as f:
        model_json = json.load(f)
    os.remove(model_file)
    merges = model_json.get("model", {}).get("merges", [])

    vocab = tokenizer.get_vocab()

    if merge_outpath:
        with open(merge_outpath, 'w', encoding='utf-8') as f:
            merges_yaml = yaml.dump(
                merges, 
                allow_unicode=True, 
                sort_keys=False)
            f.write(merges_yaml)
        print(f"Merges saved to {merge_outpath}")

    if vocab_outpath:
        with open(vocab_outpath, 'w', encoding='utf-8') as f:
            vocab_yaml = yaml.dump(
                {k: v for v, k in vocab.items()},
                allow_unicode=True,
                sort_keys=True)
            f.write(vocab_yaml)
        print(f"Vocabulary saved to {vocab_outpath}")