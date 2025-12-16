import yaml
import regex as re
from tqdm import tqdm
from array import array
from typing import Iterable, Iterator
from tokenizers import Tokenizer as HFTokenizer
from .train_bpe import GPT2_PATTERN


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        """
        Construct a tokenizer from a givenvocabulary, 
        list of merges, and (optionally) a list of special tokens. 
        """
        self.vocab = vocab
        self.merges = merges
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_pattern = f"({'|'.join(re.escape(s) for s in self.special_tokens)})"
        else:
            self.special_tokens = []
            self.special_pattern = ""

        self.encode_cache = {}
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.merges_rank = {pair: rank for rank, pair in enumerate(merges)}

        for tokens in self.special_tokens:
            token_bytes = tokens.encode("utf-8")
            if token_bytes not in self.vocab_inv:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.vocab_inv[token_bytes] = new_id

    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str, 
        merge_filepath: str, 
        special_tokens: list[str] | None = None
    ):
        """
        Class method that constructs and return a Tokenizer 
        from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) 
        and (optionally) a list of special tokens. 
        """
        with open(merge_filepath, 'r', encoding='utf-8') as f:
            merges_yaml = f.read()
            merges_list = yaml.safe_load(merges_yaml)
            merges = [(a.encode('utf-8'), b.encode('utf-8')) for a, b in merges_list]

        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_yaml = f.read()
            vocab_dict = yaml.safe_load(vocab_yaml)
            vocab = {int(k): v.encode('utf-8') for k, v in vocab_dict.items()}
        return cls(vocab, merges, special_tokens)
    
    def _encode_without_special_tokens(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs, 
        without handling special tokens.
        """
        # 'H' is the type code for uint16, which is enough for 65k vocab size
        token_ids = array('H')

        for match in GPT2_PATTERN.finditer(text):
            word = match.group(0)
            if word in self.encode_cache:
                token_ids.extend(self.encode_cache[word])
                continue

            word_bytes = [bytes([b]) for b in word.encode("utf-8")]

            while True:
                # Find the highest-ranked pair
                min_rank = float('inf')
                best_pair = None
                merged = None

                for i in range(len(word_bytes) - 1):
                    pair = (word_bytes[i], word_bytes[i+1])
                    if pair in self.merges_rank:
                        rank = self.merges_rank.get(pair, float('inf'))
                        if rank < min_rank:
                            min_rank = rank
                            best_pair = i
                            merged = pair[0] + pair[1]

                if best_pair is None:
                    break

                word_bytes = word_bytes[:best_pair] + [merged] + word_bytes[best_pair + 2:]
            
            token_id = [self.vocab_inv[token] for token in word_bytes]
            self.encode_cache[word] = token_id
            token_ids.extend(token_id)

        return token_ids.tolist()


    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        if self.special_tokens == []:
            return self._encode_without_special_tokens(text)
        
        text_parts = re.split(self.special_pattern, text)

        token_ids = []
        for part in text_parts:
            if part in self.special_tokens:
                token_ids.append(self.vocab_inv[part.encode("utf-8")])
            elif part:
                token_ids.extend(self._encode_without_special_tokens(part))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files 
        that we cannot directly load into memory.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str :
        """Decode a sequence of token IDs into text."""
        text_bytes = b"".join(self.vocab[id] for id in ids)
        return text_bytes.decode("utf-8", errors="replace")
    

class HuggingFaceTokenizer():
    def __init__(self, model_filepath: str):
        """
        Construct a HuggingFace Tokenizer from a serialized tokenizer.json file.
        """
        self.tokenizer = HFTokenizer.from_file(model_filepath)

    def encode_lines(
        self, 
        input_path: str,
        batch_lines: int = 10000,
    ) -> list[int]:
        """Encode lines from a file into a list of token IDs."""
        token_ids = []
        with open(input_path, "r", encoding="utf-8") as f:
            batch = []
            for line in tqdm(f, desc="Tokenizing lines"):
                line = line.strip()
                if not line:
                    continue
                batch.append(line)
                if len(batch) >= batch_lines:
                    encs = self.tokenizer.encode_batch(batch)
                    batch.clear()
                    for enc in encs:
                        token_ids.extend(enc.ids)
            if batch:
                encs = self.tokenizer.encode_batch(batch)
                batch.clear()
                for enc in encs:
                    token_ids.extend(enc.ids)

        return token_ids
    
