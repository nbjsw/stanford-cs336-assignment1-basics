import os
import pickle
import argparse
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.implementations import ByteLevelBPETokenizer
import json


def train_bpe_fast(input_path: str,
                   vocab_size: int,
                   special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Using Hugging Face tokenizers for BPE training.
    
    Returns:
        A tuple containing two elements:
        - vocab: dict[int, bytes] (Token ID to byte sequence)
        - merges: list[tuple[bytes, bytes]] (List of byte pair merges)
    """
    
    # 1. 初始化一个 Byte-Level BPE Tokenizer
    # ByteLevelBPE 自动处理 Unicode 字节编码，非常适合你的字节级别 BPE
    tokenizer_fast = ByteLevelBPETokenizer(
        add_prefix_space=False, # GPT-2/3 风格通常不需要前缀空格
        trim_offsets=False,
    )

    # 3. 开始训练
    # 这里的 files=[input_path] 替换了你原代码中复杂的并行读取和计数逻辑
    tokenizer_fast.train(
        files=[input_path],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
    )

    # 4. 提取和转换输出格式

    # A. 提取 Vocab
    vocab_str_to_id = tokenizer_fast.get_vocab()
    vocab_bytes: dict[int, bytes] = {}
    for token_str, token_id in vocab_str_to_id.items():
        try:
            token_bytes = token_str.encode("utf-8")
        except:
            raise RuntimeError(f"Could not convert token string {token_str} to bytes.")
        vocab_bytes[token_id] = token_bytes

    # B. 提取 Merges
    temp_json_path = "temp_tokenizer_output.json"
    tokenizer_fast.save(temp_json_path)

    with open(temp_json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    merges = config['model']['merges']
    merges_bytes: list[tuple[bytes, bytes]] = []
    for merge in merges:
        if isinstance(merge, list) and len(merge) ==2:
            p1_str, p2_str = merge[0], merge[1]
        elif isinstance(merge, str):
            p1_str, p2_str = merge.split(' ')
        else:
            raise ValueError(f"Unexpected merge format: {merge_item}")
        p1_bytes = p1_str.encode("utf-8")
        p2_bytes = p2_str.encode("utf-8")
        merges_bytes.append((p1_bytes, p2_bytes))

    os.remove(temp_json_path)
    return vocab_bytes, merges_bytes


def train_and_save_fast(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str],
    vocab_output_path: str,
    merges_output_path: str
):
    """Runs BPE training using the high-performance implementation and saves the resulting vocabulary and merges."""

    print(f"Starting BPE training on {input_path} (Using Rust-based Tokenizers library)...")
    
    vocab, merges = train_bpe_fast(input_path, vocab_size, special_tokens)
    
    print(f"Training complete. Final Vocab Size: {len(vocab)}. Total Merges: {len(merges)}")

    # 1. 保存 Vocab (使用 pickle) - 保持格式不变
    with open(vocab_output_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Saved vocabulary (dict[int, bytes]) to {vocab_output_path}")

    # 2. 保存 Merges (使用文本文件) - 保持格式不变
    save_merges_txt(merges, merges_output_path)
    print(f"Saved merges (list[tuple[bytes, bytes]]) to {merges_output_path}")


def save_merges_txt(merges: list[tuple[bytes, bytes]], filepath: str):
    """Saves the list of BPE merges to a text file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for p1, p2 in merges:
            # 使用 repr() 将 bytes 对象转换为可读的字符串表示 (如 b'\xe4' -> "\xe4")
            f.write(f"{p1!r} {p2!r}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a BPE Tokenizer and save assets.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the raw text file for BPE training (e.g., TinyStoriesV2-GPT4-train.txt).")
    parser.add_argument('--vocab_size', type=int, default=10000, help="Target final vocabulary size.")
    parser.add_argument('--special_tokens', type=str, default='<|endoftext|>', help="Comma-separated list of special tokens.")
    parser.add_argument('--vocab_path', type=str, required=True, help="Output path for the vocabulary file (.pkl).")
    parser.add_argument('--merges_path', type=str, required=True, help="Output path for the merges file (.txt).")

    args = parser.parse_args()

    special_tokens = [t.strip() for t in args.special_tokens.split(',')]
    
    train_and_save_fast(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        vocab_output_path=args.vocab_path,
        merges_output_path=args.merges_path
    )
