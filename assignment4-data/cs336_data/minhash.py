import os
import mmh3 as mmh
import regex as re
from unicodedata import normalize
from collections import defaultdict


class MinHashDeduplicator:
    def __init__(self):
        self._reset()

    def _reset(self):
        self.signatures = []
        self.candidates = []
        self.buckets = defaultdict(set)
    
    def normalize_text(self, text: str) -> str:
        text = normalize("NFD", text)
        text = re.sub(r"\p{Mn}+", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.lower()
        return text

    def get_signature(
        self,
        shingles:  set[str],
        num_hashes: int, # of hash functions
    ) -> list[int]:
        signature = [float('inf')] * num_hashes
        for shingle in shingles:
            for i in range(num_hashes):
                hash_value = mmh.hash(shingle, seed=i)
                if hash_value < signature[i]:
                    signature[i] = hash_value
        return signature

    def lsh_bands(
        self,
        doc_id,
        signature: list[int],
        num_bands: int,
    ) -> None:
        assert len(signature) % num_bands == 0, "Signature length must be divisible by number of bands"
        rows_per_band = len(signature) // num_bands
        for band in range(num_bands):
            start = band * rows_per_band
            end = start + rows_per_band
            band_signature = tuple(signature[start:end])
            self.buckets[band_signature].add(doc_id)

    def jaccard_similarity(
        self,
        set1: set,
        set2: set,
    ) -> float:
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        if union == 0:
            return 0.0
        return intersection / union

    def minhash_deduplication(
        self,
        input_files: list[os.PathLike],
        num_hashes: int,
        num_bands: int,
        ngrams: int,
        jaccard_threshold: float,
        output_directory: os.PathLike,
    ) -> None:
        # reset state
        self._reset()

        # load files, normalize, compute shingles and signatures
        docs = []
        basenames = []
        shingles_list = []
        for path in input_files:
            p = str(path)
            with open(p, 'rt') as infile:
                raw = infile.read()
            norm = self.normalize_text(raw)
            docs.append(raw)
            basenames.append(os.path.basename(p))

            # build shingles
            words = norm.split()
            if len(words) < ngrams:
                shingles = set([" ".join(words)])
            else:
                shingles = set(
                    " ".join(words[i : i + ngrams])
                    for i in range(len(words) - ngrams + 1)
                )
            shingles_list.append(shingles)

            # compute and store signature
            sig = self.get_signature(shingles, num_hashes)
            self.signatures.append(sig)

        # LSH banding
        for doc_id, signature in enumerate(self.signatures):
            self.lsh_bands(doc_id, signature, num_bands)

        # collect candidate pairs from buckets
        candidate_pairs = set()
        for bucket in self.buckets.values():
            bucket = list(bucket)
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    candidate_pairs.add((bucket[i], bucket[j]))

        # compute candidate pairs' true Jaccard similarity
        self.candidates = []
        for a, b in sorted(candidate_pairs):
            s1 = shingles_list[a]
            s2 = shingles_list[b]
            sim = self.jaccard_similarity(s1, s2)
            self.candidates.append((a, b, sim))

        # deduplication
        kept = [True] * len(docs)
        # build adjacency for deterministic iteration
        adj = defaultdict(list)
        for a, b, sim in self.candidates:
            adj[a].append((b, sim))
            adj[b].append((a, sim))

        for i in range(len(docs)):
            if not kept[i]:
                continue
            # sort candidates to ensure deterministic behavior
            for cand, sim in sorted(adj.get(i, []), key=lambda x: x[0]):
                if not kept[cand]:
                    continue
                if sim >= jaccard_threshold:
                    kept[cand] = False

        # write kept documents to output directory preserving filenames
        os.makedirs(str(output_directory), exist_ok=True)
        for idx, keep in enumerate(kept):
            if keep:
                out_path = os.path.join(str(output_directory), basenames[idx])
                with open(out_path, 'wt') as outfile:
                    outfile.write(docs[idx])
        