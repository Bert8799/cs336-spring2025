import json
import torch
import random
from dataclasses import dataclass
from typing import Dict, List, Any
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class DPODataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        shuffle: bool = False,
    ):
        self.tokenizer = tokenizer
        self.prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        self.data = self._load_data(data_path)
        
        if shuffle:
            random.shuffle(self.data)

    def _load_data(self, path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        prompt_text = self.prompt_template.format(instruction=sample["instruction"])
        
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        chosen_ids = self.tokenizer.encode(sample["chosen"], add_special_tokens=False) + [self.tokenizer.eos_token_id]
        rejected_ids = self.tokenizer.encode(sample["rejected"], add_special_tokens=False) + [self.tokenizer.eos_token_id]

        c_input_ids = prompt_ids + chosen_ids
        r_input_ids = prompt_ids + rejected_ids

        return {
            "prompt_len": len(prompt_ids),
            "chosen_input_ids": c_input_ids,
            "rejected_input_ids": r_input_ids,
        }


@dataclass
class DPODataCollator:
    pad_token_id: int
    padding_side: str = "right"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Input: List of samples from Dataset
        Output: Batch dictionary with pre-computed loss masks
        """
        chosen_ids = [torch.tensor(f["chosen_input_ids"], dtype=torch.long) for f in features]
        rejected_ids = [torch.tensor(f["rejected_input_ids"], dtype=torch.long) for f in features]
        prompt_lens = [f["prompt_len"] for f in features]

        pad_id = self.pad_token_id
        chosen_input_ids = pad_sequence(chosen_ids, batch_first=True, padding_value=pad_id)
        rejected_input_ids = pad_sequence(rejected_ids, batch_first=True, padding_value=pad_id)

        def _make_loss_mask(input_ids, p_lens):
            mask = torch.zeros_like(input_ids)
            for i, p_len in enumerate(p_lens):
                seq_len = (input_ids[i] != pad_id).sum()
                
                if seq_len > p_len:
                    mask[i, p_len:seq_len] = 1
            return mask

        chosen_loss_mask = _make_loss_mask(chosen_input_ids, prompt_lens)
        rejected_loss_mask = _make_loss_mask(rejected_input_ids, prompt_lens)

        chosen_attn_mask = chosen_input_ids.ne(pad_id).long()
        rejected_attn_mask = rejected_input_ids.ne(pad_id).long()

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attn_mask,
            "chosen_loss_mask": chosen_loss_mask,

            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attn_mask,
            "rejected_loss_mask": rejected_loss_mask,
        }