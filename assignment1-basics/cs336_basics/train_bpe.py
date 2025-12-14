import os
from typing import BinaryIO
import regex as re
import multiprocessing as mp
import time
import heapq
from collections import defaultdict


GPT2_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


class Node:
    """Represents a node in a doubly linked list for tokens."""
    def __init__(self, token: bytes):
        self.token = token
        self.prev: Node | None = None
        self.next: Node | None = None


class TokenSequence:
    """Represents a sequence of tokens as a doubly linked list."""
    def __init__(self, tokens: tuple[bytes, ...], count: int):
        self.count = count  # Frequency of this sequence
        self.head: Node | None = None
        self.tail: Node | None = None
        
        # Build the doubly linked list
        if tokens:
            self.head = Node(tokens[0])
            current = self.head
            for token in tokens[1:]:
                new_node = Node(token)
                current.next = new_node
                new_node.prev = current
                current = new_node
            self.tail = current
    
    def to_tuple(self) -> tuple[bytes, ...]:
        """Converts the linked list back to a tuple"""
        result = []
        current = self.head
        while current:
            result.append(current.token)
            current = current.next
        return tuple(result)
    
    def get_pairs(self) -> list[tuple[Node, bytes, bytes]]:
        """Gets all adjacent pairs and their positions (first node)"""
        pairs = []
        current = self.head
        while current and current.next:
            pairs.append((current, current.token, current.next.token))
            current = current.next
        return pairs


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
    >>> special_pattern = re.compile(r"<\|endoftext\|>")
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
    ctx = mp.get_context("spawn")
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


def merge_pair_in_sequence(
    seq: TokenSequence, 
    pair: tuple[bytes, bytes], 
    merged_token: bytes
) -> set[tuple[bytes, bytes]]:
    """
    Merge all occurrences of the given pair in the TokenSequence.
    Returns a set of affected pairs (both removed and newly created).
    """
    affected_pairs = set()
    current = seq.head
    
    while current and current.next:
        if (current.token, current.next.token) == pair:
            # Record the pair being removed
            affected_pairs.add(pair)
            
            # Record the affected pair on the left
            if current.prev:
                affected_pairs.add((current.prev.token, current.token))
            
            # Record the affected pair on the right
            if current.next.next:
                affected_pairs.add((current.next.token, current.next.next.token))
            
            # Perform merge: change current's token to merged_token, skip current.next
            next_node = current.next
            current.token = merged_token
            current.next = next_node.next
            if next_node.next:
                next_node.next.prev = current
            else:
                seq.tail = current
            
            # Record newly created pairs
            if current.prev:
                affected_pairs.add((current.prev.token, merged_token))
            if current.next:
                affected_pairs.add((merged_token, current.next.token))
            
            # Continue from the merged node
            # Keep current unchanged as we need to check the new current.next
        else:
            current = current.next
    
    return affected_pairs

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    merge_outpath: str = None,
    vocab_outpath: str = None,
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
    print(f"Initial vocabulary size: {len(vocab)}")
    
    # Pre-tokenize the input file to get token frequencies
    token_counts = pre_tokenize(input_path, special_tokens)
    print(f"Number of unique pre-tokens: {len(token_counts)}")

    # Convert token_counts to doubly linked list structure
    sequences: dict[tuple[bytes, ...], TokenSequence] = {}
    for token_seq, count in token_counts.items():
        sequences[token_seq] = TokenSequence(token_seq, count)
    
    # Compute initial pair counts and build index from pairs to sequences
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_sequences: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    
    for token_seq, seq in sequences.items():
        for node, token1, token2 in seq.get_pairs():
            pair = (token1, token2)
            pair_counts[pair] += seq.count
            pair_to_sequences[pair].add(token_seq)
    
    print(f"Number of unique pairs: {len(pair_counts)}")
    
    # Use max heap to store pairs with custom comparison logic
    # HeapItem compares by: count (desc), first token (desc), second token (desc)
    heap = [HeapItem(count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    
    # Record performed merges
    merges: list[tuple[bytes, bytes]] = []
    
    # Perform merge operations until target vocabulary size is reached
    num_merges = vocab_size - len(vocab)
    print(f"Performing {num_merges} merges...")
    
    merge_iteration = 0
    last_report_time = time.time()
    
    while len(vocab) < vocab_size and heap:
        # Get the pair with highest frequency from the heap
        while heap:
            item = heapq.heappop(heap)
            expected_count = item.count
            best_pair = item.pair
            
            # Check if this pair is still valid (lazy deletion)
            current_count = pair_counts.get(best_pair, 0)
            if current_count == expected_count and current_count > 0:
                break
        else:
            # Heap is empty
            break
        
        # Create the new merged token
        merged_token = best_pair[0] + best_pair[1]
        
        # Add the new token to vocabulary
        vocab[len(vocab)] = merged_token
        merges.append(best_pair)
        
        # Execute merge in all sequences containing this pair
        affected_sequences = list(pair_to_sequences[best_pair])
        
        # Collect all pairs that need to be updated
        pairs_to_update: dict[tuple[bytes, bytes], int] = defaultdict(int)
        
        for token_seq in affected_sequences:
            if token_seq not in sequences:
                continue
                
            seq = sequences[token_seq]
            
            # Execute merge in sequence and get affected pairs
            affected_pairs = merge_pair_in_sequence(seq, best_pair, merged_token)
            
            # Update pair counts
            for pair in affected_pairs:
                if pair == best_pair:
                    # This pair was removed
                    pairs_to_update[pair] -= seq.count
                elif pair[0] == merged_token or pair[1] == merged_token:
                    # Newly created pair
                    pairs_to_update[pair] += seq.count
                else:
                    # Old pair that was removed
                    pairs_to_update[pair] -= seq.count
            
            # Update the key in sequences dictionary
            new_token_seq = seq.to_tuple()
            if new_token_seq != token_seq:
                del sequences[token_seq]
                sequences[new_token_seq] = seq
                
                # Update pair_to_sequences
                for pair in list(pair_to_sequences.keys()):
                    if token_seq in pair_to_sequences[pair]:
                        pair_to_sequences[pair].discard(token_seq)
                        if not pair_to_sequences[pair]:
                            del pair_to_sequences[pair]
        
        # Batch update pair_counts and heap
        for pair, count_delta in pairs_to_update.items():
            old_count = pair_counts.get(pair, 0)
            new_count = old_count + count_delta
            
            if new_count <= 0:
                # Remove this pair
                if pair in pair_counts:
                    del pair_counts[pair]
                if pair in pair_to_sequences:
                    del pair_to_sequences[pair]
            else:
                pair_counts[pair] = new_count
                # Push updated pair to heap (lazy deletion strategy)
                heapq.heappush(heap, HeapItem(new_count, pair))
                
                # Update pair_to_sequences
                if pair[0] == merged_token or pair[1] == merged_token:
                    for token_seq, seq in sequences.items():
                        for node, token1, token2 in seq.get_pairs():
                            if (token1, token2) == pair:
                                pair_to_sequences[pair].add(token_seq)
        
        merge_iteration += 1
        
        # Report progress periodically
        current_time = time.time()
        if current_time - last_report_time >= 10:  # Report every 10 seconds
            elapsed = current_time - start_time
            progress = len(merges) / num_merges * 100
            print(f"Progress: {len(merges)}/{num_merges} merges ({progress:.1f}%), "
                  f"Elapsed: {elapsed:.1f}s, Vocab size: {len(vocab)}")
            last_report_time = current_time
    
    total_time = time.time() - start_time
    print(f"Training complete! Total time: {total_time:.2f}s")
    print(f"Final vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    
    # Save merges and vocabulary
    if merge_outpath:
        with open(merge_outpath, 'w', encoding='utf-8') as f:
            for pair in merges:
                token1 = pair[0].decode('utf-8', errors='replace')
                token2 = pair[1].decode('utf-8', errors='replace')
                f.write(f"{token1} {token2}\n")
        print(f"Merges saved to {merge_outpath}")
    
    if vocab_outpath:
        with open(vocab_outpath, 'w', encoding='utf-8') as f:
            for idx, token in sorted(vocab.items()):
                token_str = token.decode('utf-8', errors='replace')
                f.write(f"{idx}\t{token_str}\n")
        print(f"Vocabulary saved to {vocab_outpath}")
    
    return vocab, merges