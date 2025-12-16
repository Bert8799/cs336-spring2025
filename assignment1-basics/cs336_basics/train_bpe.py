import os
import time
import heapq
import yaml
import json
import regex as re
import multiprocessing as mp
from tqdm import tqdm
from typing import BinaryIO
from collections import defaultdict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel


GPT2_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


class HeapItem:
    """
    Wrapper for heap items with custom comparison logic.
    Comparison order:
    1. By frequency (descending - higher counts first)
    2. By first token bytes (descending)
    3. By second token bytes (descending)
    """
    def __init__(self, count: int, pair: tuple[bytes, bytes]):
        self.count = count
        self.pair = pair
    
    def __lt__(self, other):
        # For max heap behavior with min heap, negate count comparison
        if self.count != other.count:
            return self.count > other.count
        # If counts are equal, compare by first token (descending)
        if self.pair[0] != other.pair[0]:
            return self.pair[0] > other.pair[0]
        # If first tokens are equal, compare by second token (descending)
        return self.pair[1] > other.pair[1]
    
    def __eq__(self, other):
        return self.count == other.count and self.pair == other.pair


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    !!! Copy from pretokenization_example.py !!!
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize_chunk(
    chunk: str,
    special_pattern: re.Pattern | None = None,
) -> dict[tuple[bytes], int]:
    """
    Pre-tokenize a single chunk of text.
    special_pattern is an optional regex pattern that matches special tokens to treat as single tokens.
    Returns a dictionary mapping tokens (as bytes) to their counts.

    Example:
    >>> chunk = "Hello, world!<|endoftext|>"
    >>> special_pattern = re.compile(r"<|endoftext|>")
    >>> pre_tokenize_chunk(chunk, special_pattern)
    {
        (b'H', b'e', b'l', b'l', b'o'): 1,
        (b' ',): 1,
        (b'w', b'o', b'r', b'l', b'd', b'!'): 1
    }
    """
    token_counts: dict[tuple[bytes], int] = {}

    # Special tokens themselves are not counted
    sub_chunks = special_pattern.split(chunk) if special_pattern else [chunk]
    for sub_chunk in sub_chunks:
        for match in GPT2_PATTERN.finditer(sub_chunk):
            word = match.group(0).encode("utf-8")
            tokens_byte = tuple(bytes([b]) for b in word)
            token_counts[tokens_byte] = token_counts.get(tokens_byte, 0) + 1

    return token_counts


def pre_tokenize(
    input_path: str,
    special_tokens: list[str] | None = None,
) -> dict[tuple[bytes], int]:
    """
    Pre-tokenize the entire file at input_path.
    special_tokens is an optional list of special tokens to treat as single tokens.
    Returns a dictionary mapping tokens (as bytes) to their counts.
    """
    num_processes = mp.cpu_count()
    ctx = mp.get_context("fork")
    pool = ctx.Pool(processes=num_processes)

    special_pattern = None
    if special_tokens:
        pattern_str = "|".join(re.escape(token) for token in special_tokens)
        special_pattern = re.compile(pattern_str)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        tasks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            tasks.append((chunk, special_pattern))

        results = pool.starmap(pre_tokenize_chunk, tasks)
    
    pool.close()
    pool.join()

    # Combine results from all processes
    total_token_counts: dict[tuple[bytes], int] = {}
    for token_counts in results:
        for token, count in token_counts.items():
            total_token_counts[token] = total_token_counts.get(token, 0) + count
    
    return total_token_counts


def merge(
    token_counts: dict[tuple[bytes], int],
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes]]],
    pair: tuple[bytes, bytes]
) -> set[tuple[bytes, bytes]]:
    """
    Merge the given pair in the token_counts and update pair_counts and pair_to_tokens accordingly.
    Returns the set of affected pairs whose counts need to be updated.
    """
    affected_pairs = set()
    tokens_to_update = pair_to_tokens[pair].copy()

    for old_key in tokens_to_update:
        count = token_counts[old_key]

        # Create new token by merging the pair
        new_token = []
        i = 0
        while i < len(old_key):
            if i < len(old_key) - 1 and (old_key[i], old_key[i + 1]) == pair:
                new_token.append(old_key[i] + old_key[i + 1])
                i += 2
            else:
                new_token.append(old_key[i])
                i += 1
        new_key = tuple(new_token)

        for i in range(len(old_key) - 1):
            left_pair = (old_key[i], old_key[i + 1])
            pair_counts[left_pair] -= count
            if pair_counts[left_pair] <= 0:
                del pair_counts[left_pair]
            pair_to_tokens[left_pair].discard(old_key)
            affected_pairs.add(left_pair)

        for i in range(len(new_key) - 1):
            new_left_pair = (new_key[i], new_key[i + 1])
            pair_counts[new_left_pair] += count
            pair_to_tokens[new_left_pair].add(new_key)
            affected_pairs.add(new_left_pair)
        
        token_counts[new_key] = token_counts.get(new_key, 0) + count
    
    pair_to_tokens[pair].clear()

    return affected_pairs


def load_bpe(
    merge_path: str,
    vocab_path: str
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Load a BPE tokenizer from the given merge and vocabulary files.
    Returns the vocabulary and the list of merges.
    """
    with open(merge_path, 'r', encoding='utf-8') as f:
        merges_yaml = f.read()
        merges_list = yaml.safe_load(merges_yaml)
        merges = [(a.encode('utf-8'), b.encode('utf-8')) for a, b in merges_list]

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_yaml = f.read()
        vocab_dict = yaml.safe_load(vocab_yaml)
        vocab = {int(k): v.encode('utf-8') for k, v in vocab_dict.items()}

    return vocab, merges


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    merge_outpath: str | None = None,
    vocab_outpath: str | None = None,
    **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the input file.
    special_tokens is a list of special tokens to include in the vocabulary.
    Returns the final vocabulary and the list of merges.
    """
    start_time = time.time()

    # Initialize the vocabulary with single-byte tokens
    initial_tokens = [token.encode("utf-8") for token in special_tokens]
    initial_tokens += [bytes([i]) for i in range(256)]

    # Create vocabulary with initial tokens
    vocab = {i: token for i, token in enumerate(initial_tokens)}
    
    # Pre-tokenize the input file to get token frequencies
    current_time = time.time()
    token_counts = pre_tokenize(input_path, special_tokens)
    print(f"\nPre-tokenization time: {time.time() - current_time:.2f}s")
    print(f"Pre-tokens size: {len(token_counts)}")

    # Get the initial pairs and their counts
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes]]] = defaultdict(set)
    for token_tuple, count in token_counts.items():
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            pair_counts[pair] += count
            pair_to_tokens[pair].add(token_tuple)
    
    # Create a max-heap of pairs based on their counts
    heap: list[HeapItem] = [HeapItem(count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    # Merge tokens until reaching the desired vocabulary size
    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)
    
    current_time = time.time()
    pbar = tqdm(total=num_merges, desc="Merging tokens", unit="merge")
    while len(vocab) < vocab_size:
        if not heap:
            print("No more pairs to merge.")
            break

        # Lazy deletion: pop until we find a valid pair
        while heap:
            top_item = heapq.heappop(heap)
            current_count = pair_counts.get(top_item.pair, 0)
            if current_count == top_item.count:
                break
        else:
            print("No valid pairs left to merge.")
            break

        vocab[len(vocab)] = top_item.pair[0] + top_item.pair[1]
        merges.append(top_item.pair)

        affected_pairs = merge(token_counts, pair_counts, pair_to_tokens, top_item.pair)

        for affected_pair in affected_pairs:
            if affected_pair in pair_counts and pair_counts[affected_pair] > 0:
                heapq.heappush(heap, HeapItem(pair_counts[affected_pair], affected_pair))
        pbar.update(1)
    pbar.close()

    print(f"Merges time: {time.time() - current_time:.2f}s")

    total_time = time.time() - start_time
    print(f"Training time: {total_time:.2f}s")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    
    # Save merges and vocabulary
    if merge_outpath:
        with open(merge_outpath, 'w', encoding='utf-8') as f:
            merges_yaml = yaml.dump(
                [ 
                    [a.decode('utf-8', errors='replace'), 
                     b.decode('utf-8', errors='replace')] 
                    for a, b in merges
                ], 
                allow_unicode=True)
            f.write(merges_yaml)
        print(f"Merges saved to {merge_outpath}")
    
    if vocab_outpath:
        with open(vocab_outpath, 'w', encoding='utf-8') as f:
            vocab_yaml = yaml.dump(
                {k: v.decode('utf-8', errors='replace') for k, v in vocab.items()},
                allow_unicode=True)
            f.write(vocab_yaml)
        print(f"Vocabulary saved to {vocab_outpath}")
    
    return vocab, merges


def hf_train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    merge_outpath: str | None = None,
    vocab_outpath: str | None = None,
    output_path: str | None = None,
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
    model_file = "temp_tokenizer.json" if not output_path else output_path
    tokenizer.save(model_file)
    with open(model_file, "r", encoding="utf-8") as f:
        model_json = json.load(f)
    if not output_path:
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