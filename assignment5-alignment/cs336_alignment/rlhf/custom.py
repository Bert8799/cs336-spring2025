#!/usr/bin/env python3
"""
Split data/rlhf/raw_train.jsonl.xz into train/test jsonl.xz
from the PKU SafeRLHF-10K dataset:
https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-10K
"""

import argparse
import json
import random
import logging
from pathlib import Path

from xopen import xopen

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_process_data(input_path: Path):
    processed_data = []
    
    logger.info(f"Reading from {input_path}")
    
    with xopen(input_path, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            
            try:
                row = json.loads(line)
                prompt = row.get('prompt')
                
                if row.get('safer_response_id') == 0:
                    processed_data.append({
                        "prompt": prompt,
                        "response": row.get('response_0')
                    })

                else:
                    processed_data.append({
                        "prompt": prompt,
                        "response": row.get('response_1')
                    })
                    
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON at line {line_num}")
                continue

    return processed_data

def write_jsonl(path: Path, data: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing {len(data)} examples to {path}...")
    
    with xopen(path, "wt", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Split jsonl.xz dataset without HF datasets library.")
    parser.add_argument("--input", type=Path, default=Path("data/rlhf/raw_train.jsonl.xz"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/rlhf"))
    parser.add_argument("--train-name", type=str, default="train.jsonl.xz")
    parser.add_argument("--test-name", type=str, default="test.jsonl.xz")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    all_data = load_and_process_data(args.input)
    logger.info(f"Total valid SFT samples extracted: {len(all_data)}")

    random.seed(args.seed)
    random.shuffle(all_data)

    split_idx = int(len(all_data) * (1 - args.test_size))
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]

    write_jsonl(args.out_dir / args.train_name, train_data)
    write_jsonl(args.out_dir / args.test_name, test_data)

    logger.info("Done!")

if __name__ == "__main__":
    main()