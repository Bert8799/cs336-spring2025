import argparse
import json
import regex as re
from tqdm import tqdm
from pathlib import Path
from typing import Optional

from cs336_alignment.config import BaseConfig


config = BaseConfig()
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

    with open(config.prompt_template_path, "r", encoding="utf-8") as f:
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
                question = obj.get(config.raw_question_placeholder.strip("{}"), "").strip()
                solution = obj.get(config.raw_solution_placeholder.strip("{}"), "")
                assert solution, "Solution field is empty."
                extract_func = (_extract_boxed_answer
                                if config.extract_function == "boxed"
                                else _extract_raw_answer)
                answer = extract_func(solution)
                if is_sft:
                    prompt = prompt_r1_zero.replace(config.question_placeholder, question)
                    response = " " + solution + " </think> <answer> " + answer + " </answer>"
                    record = {"prompt": prompt, "response": response, "ground_truth": answer}
                else:
                    record = {"question": question, "answer": answer}
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    _convert(src_train, out_train)
    _convert(src_test, out_val)
    _convert(src_train, out_sft, is_sft=True)

def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets")
    parser.add_argument(
        "--dataset",
        choices=["MATH", "gsm8k"],
        default=config.dataset,
        help="Dataset to preprocess",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory for the dataset (defaults to repo data path)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if args.dataset == config.dataset:
        data_dir = (
            Path(args.data_dir)
            if args.data_dir is not None
            else repo_root / "data" / config.dataset
        )
        process_math_dataset(data_dir)


if __name__ == "__main__":
    main()

