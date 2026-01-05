from __future__ import annotations

import os
from re import M
from typing import Any



def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    from cs336_data.utils import extract_text
    return extract_text(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    from cs336_data.clasifiers import identify_language
    return identify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    from cs336_data.mask_pii import mask_emails
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    from cs336_data.mask_pii import mask_phone_numbers
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    from cs336_data.mask_pii import mask_ipv4_addresses
    return mask_ipv4_addresses(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    from cs336_data.clasifiers import clasify_nsfw
    return clasify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    from cs336_data.clasifiers import classify_hatespeech
    return classify_hatespeech(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    from cs336_data.quality_model import QualityModel
    model = QualityModel()
    model.load_model()
    return model.inference(text)


def run_gopher_quality_filter(text: str) -> bool:
    from cs336_data.clasifiers import gopher_quality_filter
    return gopher_quality_filter(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    from cs336_data.utils import extract_line_deduplication
    return extract_line_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    from cs336_data.minhash import MinHashDeduplicator
    deduplicator = MinHashDeduplicator()
    deduplicator.minhash_deduplication(
        input_files,
        num_hashes,
        num_bands,
        ngrams,
        jaccard_threshold,
        output_directory,
    )
