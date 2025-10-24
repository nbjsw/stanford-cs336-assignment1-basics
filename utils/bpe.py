import multiprocessing
import os
import regex as re
from collections import defaultdict
from typing import Any
from tqdm import tqdm

# ----------------------------------------------------------------------
# Core BPE Training Functions
# ----------------------------------------------------------------------

# BPE pre-tokenization regex pattern
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def read_text_iter(input_path: str, special_tokens: list[str], chunk_size_lines: int = 1000) -> Any:
    """
    Reads the file line by line and yields chunks of lines as strings.
    This acts as a generator to avoid loading the entire file into memory.

    Args:
        input_path: The path to the input file (str).

    Returns:
        The entire content of the file as a string (str).

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
    # Core: Encode the word into a list of bytes, then convert to a tuple of single-byte objects
    return tuple(bytes([i]) for i in word.encode('utf-8'))


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
        # Following original code logic, only count words with length >= 2
        if len(word_bytes) >= 2:
            word_cnt[word_bytes] += 1
    return word_cnt


def merge_dicts(dicts: list[dict[Any, int]]) -> dict[Any, int]:
    """
    Merges multiple dictionaries with the same key type and integer value type.

    Args:
        dicts: A list of dictionaries to be merged (list[dict[Any, int]]).

    Returns:
        The merged dictionary, with keys of type Any and integer frequencies (dict[Any, int]).
    """
    merged: dict[Any, int] = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            merged[k] += v
    return merged


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


def get_max_pair(pair_cnt: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    """
    Finds the byte pair with the maximum frequency.
    Lexicographical order of the byte pair is used as a tie-breaker.

    Args:
        pair_cnt: A dictionary mapping byte pairs to their frequencies (dict[tuple[bytes, bytes], int]).

    Returns:
        The byte pair with the maximum frequency (tuple[bytes, bytes]).
        
    Raises:
        ValueError: If pair_cnt is empty.
    """
    # The max() key=(count, pair_tuple) ensures sorting first by frequency, then by lexicographical order
    if not pair_cnt:
        raise ValueError("The pair_cnt dictionary is empty, cannot find the maximum pair.")
    max_pair, _ = max(pair_cnt.items(), key=lambda x: (x[1], x[0]))
    return max_pair


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


def apply_merge(word_bytes: tuple[bytes, ...], merge: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """
    Merges all matching (A, B) pairs into (AB) within a word byte sequence.
    Note: This is a greedy, non-overlapping merge process performed from left to right.

    Args:
        word_bytes: The word byte sequence to be merged (tuple[bytes, ...]).
        merge: The merge pair (A, B) to be applied in this iteration (tuple[bytes, bytes]).

    Returns:
        The new merged word byte sequence (tuple[bytes, ...]).
    """
    # The new byte sequence resulting from the merge
    merged: bytes = merge[0] + merge[1]
    i = 0
    new_word_bytes: list[bytes] = []
    
    # Iterate and check for matches
    while i < len(word_bytes):
        # Check if the current merge pair matches
        if i < len(word_bytes) - 1 and word_bytes[i] == merge[0] and word_bytes[i + 1] == merge[1]:
            new_word_bytes.append(merged)
            i += 2  # Skip the next element
        else:
            new_word_bytes.append(word_bytes[i])
            i += 1
            
    return tuple(new_word_bytes)


def update_cnt_optimized(
    word_cnt: dict[tuple[bytes, ...], int],
    pair_cnt: dict[tuple[bytes, bytes], int],
    merge_pair: tuple[bytes, bytes]
) -> tuple[dict[tuple[bytes, ...], int], dict[tuple[bytes, bytes], int]]:
    """
    Performance-optimized version: Only updates the counts for words that contain the `merge_pair`.
    
    Args:
        word_cnt: The current words and their frequencies (dict[tuple[bytes, ...], int]).
        pair_cnt: The current adjacent byte pair frequencies (dict[tuple[bytes, bytes], int]).
        merge_pair: The merge pair (A, B) to be executed in this round (tuple[bytes, bytes]).
    
    Returns:
        A tuple containing two elements:
        - new_word_cnt: The updated words and their frequencies (dict[tuple[bytes, ...], int]).
        - new_pair_cnt: The updated adjacent byte pair frequencies (dict[tuple[bytes, bytes], int]).
    """
    
    new_word_cnt: dict[tuple[bytes, ...], int] = defaultdict(int)
    # Copy the current pair_cnt
    new_pair_cnt: dict[tuple[bytes, bytes], int] = defaultdict(int, pair_cnt)

    words_to_update: dict[tuple[bytes, ...], int] = {}
    
    # ----------------------------------------------------------------
    # Step 1: Filter words containing the merge_pair and copy unchanged words
    # ----------------------------------------------------------------
    for word_bytes, cnt in word_cnt.items():
        has_merge = False
        if len(word_bytes) >= 2:
            # Quick check for the presence of the merge_pair
            for i in range(len(word_bytes) - 1):
                if (word_bytes[i], word_bytes[i+1]) == merge_pair:
                    has_merge = True
                    break
        
        if has_merge:
            words_to_update[word_bytes] = cnt
        else:
            # Copy directly to the new word count dictionary (these words remain unchanged)
            new_word_cnt[word_bytes] += cnt

    # ----------------------------------------------------------------
    # Step 2: Update the words containing merge_pair and their corresponding pair_cnt
    # ----------------------------------------------------------------
    for word_bytes, cnt in words_to_update.items():
        # 1. Find all old pairs
        old_pairs = list(zip(word_bytes[:-1], word_bytes[1:]))

        # 2. Subtract the count of old pairs from pair_cnt
        for pair in old_pairs:
            new_pair_cnt[pair] -= cnt
            # Clean up entries with a count of 0
            if new_pair_cnt[pair] == 0:
                del new_pair_cnt[pair]

        # 3. Calculate the new word
        new_word = apply_merge(word_bytes, merge_pair)
        new_word_cnt[new_word] += cnt

        # 4. Find all new pairs in the new word and update pair_cnt
        new_pairs = list(zip(new_word[:-1], new_word[1:]))
        for p in new_pairs:
            new_pair_cnt[p] += cnt
            
    return new_word_cnt, new_pair_cnt


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    BPE Training Main Function (Single-Process Optimized Version V2).
    Trains the BPE vocabulary and merge rules based on the input text.

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

    # Lazy text generator to void OOM
    chunks_generator = read_text_iter(input_path, special_tokens=special_tokens, chunk_size_lines=10000)
    with multiprocessing.Pool(processes=num_cpus) as pool:
        # pool.imap_unordered 可以惰性地返回结果，并保持 tqdm 进度条的响应性
        # word_dicts_iterator 包含了来自所有进程的 word 频率字典
        # Calculate initial word frequency for each chunk (map replaces process_map)
        word_dicts_iterator = pool.imap_unordered(count_word, chunks_generator)
        
        # 使用 tqdm 包装迭代器，以便在结果返回时显示进度
        word_dicts: list[dict[tuple[bytes, ...], int]] = list(
            tqdm(word_dicts_iterator, desc="Counting Words (Parallel)")
        )

    # Merge word frequencies from all chunks
    word_cnt = merge_dicts(word_dicts)
    # Calculate initial adjacent byte pair frequencies
    pair_cnt = count_pair(word_cnt)

    vocab = get_basic_vocab(special_tokens)
    base_vocab_size = len(vocab)
    n_merges = vocab_size - base_vocab_size

    merges: list[tuple[bytes, bytes]] = []
    
    # Main training loop: Perform n_merges rounds
    pbar = tqdm(range(n_merges), desc="BPE Merging", unit="merge")
    for i in pbar:
        # 1. Find the pair with the maximum frequency
        if not pair_cnt:
            # If there are no more pairs to merge, stop
            break
            
        try:
            max_pair = get_max_pair(pair_cnt)
        except ValueError:
             # In the context of the for loop, if pair_cnt is empty, get_max_pair will raise, so break
             break
        
        # 2. Update the vocabulary and merges list
        new_token_id = base_vocab_size + i
        vocab[new_token_id] = max_pair[0] + max_pair[1]
        merges.append(max_pair)
        
        # 3. Core optimization: Use the exact update function to refresh counts
        word_cnt, pair_cnt = update_cnt_optimized(word_cnt, pair_cnt, max_pair)
            
    return vocab, merges
