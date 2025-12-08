import os
import multiprocessing
import regex as re
from collections import defaultdict
from typing import Any
from tqdm import tqdm
from bpe_rust import bpe_merge_loop


# ----------------------------------------------------------------------
# Core BPE Functions
# ----------------------------------------------------------------------

# BPE pre-tokenization regex pattern
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def split_by_special(text: str, special_tokens: list[str], drop_special: bool = True) -> list[str]:
    """
    Splits the text using special tokens and returns a list of non-special token chunks.
    
    Args:
        text: The input text to be split (str).
        special_tokens: A list of user-defined special tokens (list[str]).
        drop_special: Whether to discard the special tokens (bool). If False, 
                      special tokens are retained as separate elements in the result.

    Returns:
        A list of text chunks (list[str]).
    """
    if not special_tokens:
        return [text]

    # Prioritize matching longer special tokens
    special_tokens = sorted(special_tokens, key=len, reverse=True)

    # Build the regex pattern
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if not drop_special:
        # Capture special tokens if they are not to be dropped
        pattern = f"({pattern})"

    compiled_pattern = re.compile(pattern)
    chunks = compiled_pattern.split(text)
    # Filter out empty strings
    return [c for c in chunks if c]


def word2bytes(word: str) -> tuple[bytes, ...]:
    """
    Encodes a word string into UTF-8 bytes and represents it as a tuple of single-byte objects.
    
    Args:
        word: The input word string (str).

    Returns:
        A tuple consisting of single-byte 'bytes' objects (tuple[bytes, ...]).
    """
    return tuple(bytes([i]) for i in word.encode('utf-8'))


# ----------------------------------------------------------------------
# BPE Training Functions
# ----------------------------------------------------------------------

def read_text_iter(input_path: str, special_tokens: list[str], chunk_size_lines: int = 1000) -> Any:
    """
    Reads the file line by line and yields chunks of lines as strings.
    This acts as a generator to avoid loading the entire file into memory.

    Args:
        input_path: The path to the input file (str).
        special_tokens: A list of special tokens to split by (list[str]).
        chunk_size_lines: Number of lines to process at once (int).

    Yields:
        Text chunks as strings.

    Raises:
        FileNotFoundError: If the file path is invalid.
        IOError: If the file fails to read.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        chunk_lines = []
        for i, line in enumerate(f):
            chunk_lines.append(line)
            if (i + 1) % chunk_size_lines == 0:
                text_chunk = "".join(chunk_lines)
                sub_chunks = split_by_special(text_chunk, special_tokens, drop_special=True)
                for sub_chunk in sub_chunks:
                    yield sub_chunk
                chunk_lines = []

        if chunk_lines:
            text_chunk = "".join(chunk_lines)
            sub_chunks = split_by_special(text_chunk, special_tokens, drop_special=True)
            for sub_chunk in sub_chunks:
                yield sub_chunk


def count_word(text: str) -> dict[tuple[bytes, ...], int]:
    """
    Tokenizes a pre-tokenized text chunk and counts the frequency of each word byte sequence.

    Args:
        text: A pre-tokenized text chunk (str).

    Returns:
        A dictionary mapping word byte sequences to their frequencies (dict[tuple[bytes, ...], int]).
    """
    word_cnt: dict[tuple[bytes, ...], int] = defaultdict(int)
    for m in PAT.finditer(text):
        word = m.group(0)
        word_bytes = word2bytes(word)
        # Only count words with length >= 2
        if len(word_bytes) >= 2:
            word_cnt[word_bytes] += 1
    return word_cnt


def count_pair(word_cnt: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """
    Counts the frequency of all adjacent byte pairs.

    Args:
        word_cnt: A dictionary mapping word byte sequences to their frequencies (dict[tuple[bytes, ...], int]).

    Returns:
        A dictionary mapping byte pairs to their frequencies (dict[tuple[bytes, bytes], int]).
    """
    pair_cnt: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for word_bytes, cnt in word_cnt.items():
        # Iterate over all adjacent pairs in a word byte sequence
        for i in range(len(word_bytes) - 1):
            pair_cnt[(word_bytes[i], word_bytes[i+1])] += cnt
    return pair_cnt


def get_basic_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """
    Builds the basic vocabulary: includes 256 fundamental UTF-8 bytes and user-defined special tokens.

    Args:
        special_tokens: A list of user-defined special token strings (list[str]).

    Returns:
        The basic vocabulary (dict[int, bytes]), mapping token ID to the corresponding byte sequence.
    """
    # 256 base byte token IDs -> single-byte 'bytes' objects
    vocab: dict[int, bytes] = {token: bytes([token]) for token in range(256)}
    # Special tokens, starting from ID 256
    for i, token in enumerate(special_tokens):
        token_id = 256 + i
        vocab[token_id] = token.encode("utf-8")
    return vocab


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    BPE Training Main Function (Rust-accelerated version).
    
    Args:
        input_path: The path to the text file for training (str).
        vocab_size: The target size of the final vocabulary (int).
        special_tokens: A list of user-defined special tokens (list[str]).

    Returns:
        A tuple containing two elements:
        - vocab: The final vocabulary (dict[int, bytes]), mapping token ID to the corresponding byte sequence.
        - merges: The list of BPE merge steps (list[tuple[bytes, bytes]]).
        
    Raises:
        FileNotFoundError/IOError: If the file fails to read.
    """
    num_cpus = os.cpu_count() or 4
    print(f"Using {num_cpus} processes for pre-tokenization.")

    # Lazy text generator to avoid OOM
    chunks_generator = read_text_iter(input_path, special_tokens=special_tokens, chunk_size_lines=10000)
    word_cnt = defaultdict(int)
    with multiprocessing.Pool(processes=num_cpus) as pool:
        for d in tqdm(pool.imap_unordered(count_word, chunks_generator), desc="Counting Words (Parallel)"):
            for k, v in d.items():
                word_cnt[k] += v

    # Calculate initial adjacent byte pair frequencies
    pair_cnt = count_pair(word_cnt)

    vocab = get_basic_vocab(special_tokens)
    base_vocab_size = len(vocab)
    n_merges = vocab_size - base_vocab_size

    # Use Rust-accelerated merge loop
    merges_list = bpe_merge_loop(dict(word_cnt), dict(pair_cnt), n_merges)
    
    # Convert to required format and update vocab
    merges: list[tuple[bytes, bytes]] = []
    for i, merge_tuple in enumerate(tqdm(merges_list, desc="Building vocab")):
        merge_pair = (merge_tuple[0], merge_tuple[1])
        merges.append(merge_pair)
        new_token_id = base_vocab_size + i
        vocab[new_token_id] = merge_pair[0] + merge_pair[1]
            
    return vocab, merges