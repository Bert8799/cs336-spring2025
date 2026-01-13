import json
import regex as re
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List

from cs336_alignment.config import BaseConfig


pre_cfg = BaseConfig()
pattern = r'\\frac\{(\d+)\}(\d)'
fixed = r'\\frac{\1}{\2}'

def _extract_boxed_answer(solution: str) -> Optional[str]:
    """Extract the content of the last \boxed{...} from a LaTeX solution string.

    Handles nested braces by counting. Returns None if no boxed is found.
    """
    if not solution:
        return None

    marker = "\\boxed"
    last_idx = solution.rfind(marker)
    if last_idx == -1:
        return None

    # Start after the opening brace of \boxed{
    i = last_idx + len(marker)
    end = ["$", "."]
    if solution[i] == "{":
        i += 1
        depth = 1
    else:
        depth = 0
    content_chars = []
    if solution[i] == ".":
        content_chars.append(".")
        i += 1
    while i < len(solution) and solution[i] not in end:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
            if depth == 0:
                break
        content_chars.append(solution[i])
        i += 1
    assert content_chars, "No content found inside {solution}"
    answer = "".join(content_chars).strip()
    matched = re.match(pattern, answer)
    if matched:
        answer = re.sub(pattern, fixed, answer)
    return answer


def _extract_raw_answer(solution: str) -> Optional[str]:
    """Extract the final answer from a solution string.

    Looks for patterns like '### ...'.
    Returns None if no final answer is found.
    """
    pattern = r"####\s*(-?[\d,]+(?:\.\d+)?)"
    matches = re.findall(pattern, solution)
    if matches:
        return matches[-1].strip()
    return None


def filter_long_data(input_file: Path, output_file: Path, percentile: float = 0.9) -> None:
    """Filter out data that exceeds a certain length percentile.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        percentile: Percentile threshold (default 0.9 for 90th percentile)
    """
    # First pass: collect all data and their lengths
    data_list: List[dict] = []
    lengths: List[int] = []
    
    with input_file.open("r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Reading data"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data_list.append(obj)
            # Calculate length (you can adjust this based on which field to measure)
            # Here we measure the total length of the JSON string
            length = len(json.dumps(obj, ensure_ascii=False))
            lengths.append(length)
    
    # Calculate the percentile threshold
    if not lengths:
        print("No data found in input file.")
        return
    
    lengths_sorted = sorted(lengths)
    threshold_idx = int(len(lengths_sorted) * percentile)
    threshold_length = lengths_sorted[threshold_idx]
    
    print(f"Total data entries: {len(data_list)}")
    print(f"{percentile*100}th percentile length: {threshold_length}")
    
    # Second pass: write only data below threshold
    kept_count = 0
    with output_file.open("w", encoding="utf-8") as fout:
        for obj, length in zip(data_list, lengths):
            if length <= threshold_length:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept_count += 1
    
    print(f"Kept {kept_count}/{len(data_list)} entries ({kept_count/len(data_list)*100:.2f}%)")
    print(f"Removed {len(data_list) - kept_count} entries")


def process_math_dataset(data_dir: Path) -> None:
    """Process MATH dataset JSONL files, extracting problem and boxed answer.

    Reads train_raw.jsonl and test_raw.jsonl from data_dir and writes
    train.jsonl and validation.jsonl containing objects with fields:
      - problem: original problem text
      - answer: content from the solution
    """
    src_train = data_dir / "train_raw.jsonl"
    src_test = data_dir / "test_raw.jsonl"
    out_train = data_dir / "train.jsonl"
    out_val = data_dir / "validation.jsonl"
    out_sft = data_dir / "sft.jsonl"

    with open(pre_cfg.prompt_template_path, "r", encoding="utf-8") as f:
        prompt_r1_zero = f.read()

    if not src_train.exists() or not src_test.exists():
        raise FileNotFoundError(
            f"Expected raw files at {src_train} and {src_test}, but one or both are missing."
        )

    def _convert(src: Path, dest: Path, is_sft: bool=False) -> None:
        with src.open("r", encoding="utf-8") as fin, dest.open(
            "w", encoding="utf-8"
        ) as fout:
            for line in tqdm(fin):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                question = obj.get(pre_cfg.raw_question_placeholder.strip("{}"), "").strip()
                solution = obj.get(pre_cfg.raw_solution_placeholder.strip("{}"), "")
                assert solution, "Solution field is empty."
                extract_func = (_extract_boxed_answer
                                if pre_cfg.extract_function == "boxed"
                                else _extract_raw_answer)
                answer = extract_func(solution)
                if is_sft:
                    prompt = prompt_r1_zero.replace(pre_cfg.question_placeholder, question)
                    response = " " + solution + " </think> <answer> " + answer + " </answer>"
                    record = {"prompt": prompt, "response": response, "ground_truth": answer}
                else:
                    record = {"prompt": question, "ground_truth": answer}
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    _convert(src_train, out_train)
    _convert(src_test, out_val)
    _convert(src_train, out_sft, is_sft=True)

    filter_long_data(out_train, out_train, percentile=pre_cfg.filter_long_ratio)
    filter_long_data(out_val, out_val, percentile=pre_cfg.filter_long_ratio)
    filter_long_data(out_sft, out_sft, percentile=pre_cfg.filter_long_ratio)

def main():
    data_dir = Path(pre_cfg.data_dir) / pre_cfg.dataset
    process_math_dataset(data_dir)


if __name__ == "__main__":
    main()

