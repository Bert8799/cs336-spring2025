import os
import mmh3
import gzip
import random
from clasifiers import *
from collections import Counter
from fastwarc import ArchiveIterator, WarcRecordType
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text


def extract_text(html_bytes: bytes) -> str | None:
    encoding = detect_encoding(html_bytes)
    html_str = html_bytes.decode(encoding, errors="replace")
    text = extract_plain_text(html_str)
    return text


def extract_line_deduplication(input_files: str, output_directory: str) -> None:
    counter = Counter()
    for input_file in input_files:
        with open(input_file, 'rt') as infile:
            for line in infile:
                line = line.rstrip('\n')
                hash_value = mmh3.hash(line)
                counter[hash_value] += 1
    for input_file in input_files:
        base_name = os.path.basename(input_file)
        output_path = os.path.join(output_directory, base_name)
        with open(input_file, 'rt') as infile, open(output_path, 'wt') as outfile:
            for line in infile:
                line = line.rstrip('\n')
                hash_value = mmh3.hash(line)
                if counter[hash_value] == 1:
                    outfile.write(line + '\n')


def sample_positive_urls(
    input_path='../data/wiki/enwiki-20240420-extracted_urls.txt',
    output_path='../data/wiki/10k-positive_urls.txt',
    sample_size=10_000,
) -> None:
    random.seed(42)
    reservoir = []

    with open(input_path, 'r') as infile:
        for i, line in enumerate(infile):
            url = line.strip()
            if i < sample_size:
                reservoir.append(url)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    reservoir[j] = url
            
            if (i + 1) % 100_000 == 0:
                print(f"\rProcessed {i + 1} URLs", end='', flush=True)

    with open(output_path, 'wt') as outfile:
        for url in reservoir:
            outfile.write(url + '\n')


def filter_positive_data(input_path: str, output_path: str, sample_size: int = 1000) -> None:
    with gzip.open(input_path, 'rb') as infile, open(output_path, 'wt') as outfile:
        i = 0
        for record in ArchiveIterator(infile):
            if record.record_type == WarcRecordType.response and record.content_length > 0:
                html_bytes = record.reader.read()
                text = extract_text(html_bytes)

                label, confidence = identify_language(text)
                if label != 'en' or confidence < 0.9:
                    continue

                label, confidence = clasify_nsfw(text)
                if label == 'nsfw' and confidence >= 0.9:
                    continue

                label, confidence = classify_hatespeech(text)
                if label == 'hatespeech' and confidence >= 0.9:
                    continue

                if not gopher_quality_filter(text):
                    continue

                sample = text.replace('\n', ' ') + "\n"
                outfile.write("__label__positive " + sample)
                i += 1
                print(f"\rProcessed {i} records.", end='')
                if i == sample_size:
                    break
        print(f"\nTotal processed records: {i}")


def sample_negative_data(input_path: str, output_path: str, sample_size: int = 1000) -> None:
    random.seed(42)
    reservoir = []
    with gzip.open(input_path, 'rb') as infile:
        j = 0
        for i, record in enumerate(ArchiveIterator(infile)):
            if record.record_type == WarcRecordType.response and record.content_length > 0:
                html_bytes = record.reader.read()
                text = extract_text(html_bytes)
                if text is None:
                    continue
                label, confidence = identify_language(text)
                if label != 'en' or confidence < 0.9:
                    continue

                if j < sample_size:
                    reservoir.append(text)
                    j += 1
                else:
                    j = random.randint(0, i)
                    if j < sample_size:
                        reservoir[j] = text

            print(f"\rProcessed {i + 1} records", end='', flush=True)

    print(f'\n Writing {len(reservoir)} samples to {output_path}')
    with open(output_path, 'wt') as outfile:
        for data in reservoir:
            sample = data.replace('\n', ' ') + "\n"
            outfile.write("__label__negative " + sample)


def merge_datasets(postive: str, negative: str, output_path: str) -> None:
    with open(postive, 'rt') as pos_file,  \
         open(negative, 'rt') as neg_file, \
         open(output_path, 'wt') as outfile:
        length = min(
            sum(1 for _ in pos_file),
            sum(1 for _ in neg_file)
        )
        pos_file.seek(0)
        neg_file.seek(0)
        for _ in range(length):
            pos_line = pos_file.readline()
            neg_line = neg_file.readline()
            outfile.write(pos_line)
            outfile.write(neg_line)

