import json
import torch
import random
from os import PathLike
from pathlib import Path
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from xopen import xopen


class SFTDataset(Dataset):
    def __init__(
        self, 
        tokenizer: AutoTokenizer, 
        dataset_path: PathLike | str, 
        seq_length: int,
        shuffle: bool
    ):
        """
        seq_length is the desired length of sequences to generate from the dataset 
        (typically the desired language model context length)
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.shuffle = shuffle
        
        raw_data = self._load_data(dataset_path)

        prompt_file_path = Path('cs336_alignment') / 'prompts' / 'alpaca_sft.prompt'
        with open(prompt_file_path, 'r') as f:
            self.prompt_template = f.read()
        
        self.input_ids, self.labels = self._process_data(raw_data, shuffle)

    
    def _load_data(self, path: PathLike | str):
        path = str(path)
        assert path.endswith('.jsonl') or path.endswith('.jsonl.gz') or path.endswith('.jsonl.xz'), \
            "Dataset file must be .jsonl, .jsonl.gz, or .jsonl.xz"

        with xopen(path, 'rt', encoding='utf-8') as f:
            return [json.loads(line.strip()) for line in f]
    
    def _process_data(self, data: list[dict], shuffle: bool) -> tuple[list[list[int]], list[list[int]]]:
        prompts = [
            self.prompt_template.format(
                instruction=item['prompt'],
                response=item['response']
            ) for item in data
        ]

        if shuffle:
            random.shuffle(prompts)

        encoded = self.tokenizer(prompts, add_special_tokens=False)["input_ids"]

        # Some tokenizers may not define BOS/EOS token ids.
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # Qwen3-1.7B: bos_token_id may be None in some tokenizer configs.
        bos = [bos_id] if bos_id is not None else [151644] # <|im_start|>
        eos = [eos_id] if eos_id is not None else [151645] # <|im_end|>
        
        all_list = list(chain.from_iterable((bos + ids + eos) for ids in encoded))

        # Drop last
        total_tokens = len(all_list) - 1
        trunc_len = (total_tokens // self.seq_length) * self.seq_length
        input_ids = [all_list[i:i + self.seq_length] for i in range(0, trunc_len, self.seq_length)]
        labels = [all_list[i:i + self.seq_length] for i in range(1, trunc_len + 1, self.seq_length)]
        return input_ids, labels

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        tokens = self.input_ids[idx]
        labels = self.labels[idx]

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels
        }


def iterate_batches(dataset: Dataset, batch_size: int, shuffle: bool = False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # asynchronous data loading to GPU
        pin_memory=True,
        # SFTDataset has already handled the length alignment
        # drop_last=True,
    )