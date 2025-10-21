import numpy as np
import os
import pickle
import argparse

from utils import bpe, tokenizer


def save_merges_txt(merges: list[tuple[bytes, bytes]], filepath: str):
    """Saves the list of BPE merges to a text file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for p1, p2 in merges:
            # 使用 repr() 将 bytes 对象转换为可读的字符串表示 (如 b'\xe4' -> "\xe4")
            # 方便读取和调试。
            f.write(f"{p1!r} {p2!r}\n")

def train_and_save(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str],
    vocab_output_path: str,
    merges_output_path: str
):
    """Runs BPE training and saves the resulting vocabulary and merges."""
    
    print(f"Starting BPE training on {input_path}...")
    
    # 调用你 bpe.py 中的核心训练函数
    vocab, merges = bpe.train_bpe(input_path, vocab_size, special_tokens)
    
    print(f"Training complete. Final Vocab Size: {len(vocab)}. Total Merges: {len(merges)}")

    # 1. 保存 Vocab (使用 pickle)
    with open(vocab_output_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Saved vocabulary (dict[int, bytes]) to {vocab_output_path}")

    # 2. 保存 Merges (使用文本文件)
    save_merges_txt(merges, merges_output_path)
    print(f"Saved merges (list[tuple[bytes, bytes]]) to {merges_output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a BPE Tokenizer and save assets.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the raw text file for BPE training (e.g., TinyStoriesV2-GPT4-train.txt).")
    parser.add_argument('--vocab_size', type=int, default=10000, help="Target final vocabulary size.")
    parser.add_argument('--special_tokens', type=str, default='<|endoftext|>', help="Comma-separated list of special tokens.")
    parser.add_argument('--vocab_path', type=str, required=True, help="Output path for the vocabulary file (.pkl).")
    parser.add_argument('--merges_path', type=str, required=True, help="Output path for the merges file (.txt).")
    
    args = parser.parse_args()
    
    special_tokens = [t.strip() for t in args.special_tokens.split(',')]
    
    train_and_save(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        vocab_output_path=args.vocab_path,
        merges_output_path=args.merges_path
    )

