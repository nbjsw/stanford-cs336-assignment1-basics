from typing import Iterator, Iterable, Union, Optional, Any
import json
import pickle
import regex as re
import ast

from . import bpe


def split_to_words(text: str) -> list[str]:
    """Split text into words using the bpe.PAT regex."""
    return bpe.PAT.findall(text)


def apply_merges(
    word_bytes: tuple[bytes, ...],
    merges_set: set[tuple[bytes, bytes]],
    vocab_to_id: dict[bytes, int]
) -> tuple[bytes, ...]:
    """
    Apply BPE merges iteratively to a list of token bytes.
    The merge with the lowest resulting token ID is prioritized (greedy approach).
    """
    word_bytes_list: list[bytes] = list(word_bytes)

    while True:
        min_token_id: float = float('inf')
        best_pair_idx: int = -1
        merged: Optional[bytes] = None

        for i in range(len(word_bytes_list) - 1):
            pair: tuple[bytes, bytes] = (word_bytes_list[i], word_bytes_list[i + 1])
            if pair in merges_set:
                combined: bytes = pair[0] + pair[1]
                token_id: Optional[int] = vocab_to_id.get(combined)
                
                if token_id is not None and token_id < min_token_id:
                    min_token_id = token_id
                    best_pair_idx = i
                    merged = combined

        if best_pair_idx == -1:
            break

        # Apply best merge
        if merged is not None:
            word_bytes_list = (
                word_bytes_list[:best_pair_idx]
                + [merged]
                + word_bytes_list[best_pair_idx + 2:]
            )

    return tuple(word_bytes_list)


def encode_merged(
    text: str, 
    merges: set[tuple[bytes, bytes]], 
    vocab_to_id: dict[bytes, int]
) -> list[int]:
    """
    Encode a text chunk (without special tokens) using word splitting and BPE merging.
    """
    word_list: list[str] = split_to_words(text)
    tokens: list[int] = []
    
    for word in word_list:
        word_bytes: tuple[bytes, ...] = bpe.word2bytes(word)
        merged_word_bytes: tuple[bytes, ...] = apply_merges(word_bytes, merges, vocab_to_id)
        
        tokens.extend(vocab_to_id[i] for i in merged_word_bytes)
        
    return tokens



def load_merges_txt(filepath: str) -> list[tuple[bytes, bytes]]:
    """Loads the ordered list of BPE merges from a text file."""
    merges: list[tuple[bytes, bytes]] = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            # 格式示例: b'th' b'e'
            parts = line.split()
            if len(parts) == 2:
                try:
                    p1 = ast.literal_eval(parts[0])
                    p2 = ast.literal_eval(parts[1])
                    if isinstance(p1, bytes) and isinstance(p2, bytes):
                        merges.append((p1, p2))
                except Exception as e:
                    print(f"Warning: Could not parse merge line: {line}. Error: {e}")

    return merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab: dict[int, bytes] = vocab
        self.merges: set[tuple[bytes, bytes]] = set(merges)
        self.special_tokens: list[str] = special_tokens if special_tokens else []
        self.special_tokens_bytes: list[bytes] = [i.encode('utf-8') for i in self.special_tokens]
        self.vocab_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}

        for token_bytes in self.special_tokens_bytes:
            if token_bytes not in self.vocab_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.vocab_to_id[token_bytes] = new_id

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[list[str]] = None) -> 'Tokenizer':
        """Load tokenizer data from vocab and merges files."""

        with open(vocab_filepath, 'rb') as f:
            vocab: dict[int, bytes]  = pickle.load(f)
                
        with open(merges_filepath, 'r', encoding='utf-8') as mf:
            lines: list[str] = mf.readlines()
            merge_pairs: list[tuple[str, str]] = []
            
            for line in lines:
                stripped_line: str = line.strip()
                if not stripped_line.startswith('#') and stripped_line:
                    parts: list[str] = stripped_line.split()
                    if len(parts) == 2:
                        merge_pairs.append((parts[0], parts[1]))

            merges: list[tuple[bytes, bytes]] = [(a.encode('utf-8'), b.encode('utf-8')) for a, b in merge_pairs]

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        chunks: list[str] = bpe.split_by_special(text, self.special_tokens, drop_special=False)
        tokens: list[int] = []

        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(self.vocab_to_id[chunk.encode('utf-8')])
            else:
                tokens.extend(encode_merged(chunk, self.merges, self.vocab_to_id))
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        token_bytes_list: list[bytes] = [self.vocab[t] for t in ids]
        full_bytes: bytes = b''.join(token_bytes_list)
        return full_bytes.decode('utf-8', errors='replace')

