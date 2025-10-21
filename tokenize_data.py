import numpy as np
import argparse

from tqdm import tqdm
from utils import tokenizer


def tokenize_and_save(
    raw_text_path: str, 
    vocab_path: str, 
    merges_path: str, 
    output_path: str,
    special_tokens: list[str],
    dtype: str = 'uint16'
):
    """
    Loads Tokenizer, tokenizes raw text, and saves the result as a NumPy array.
    """
    print("--- Starting Tokenization ---")
    
    # 1. 加载 Tokenizer
    tokenizer_instance = tokenizer.Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    
    print(f"Tokenizer initialized. Final Vocab Size: {len(tokenizer_instance.vocab)}")

    # 2. 内存高效地打开原始文本文件
    print(f"Tokenizing raw text from {raw_text_path}...")
    token_ids_list = []
    
    # 使用 open() 函数作为字符串的可迭代对象
    with open(raw_text_path, 'r', encoding='utf-8') as f:
        # 使用 encode_iterable 方法来惰性地处理文本流
        token_stream = tokenizer_instance.encode_iterable(f)
        
        # 将生成的 token IDs 收集到一个列表中
        for token_id in tqdm(token_stream, unit=' token', desc='Tokenizing Text'):
            token_ids_list.append(token_id)

    print(f"Tokenization complete. Total tokens: {len(token_ids_list)}")
    
    # 3. 转换为 NumPy 数组并选择合适的 dtype
    token_array = np.array(token_ids_list, dtype=dtype)
    
    # 4. 保存为 .npy 文件 (用于训练循环中的 np.memmap)
    np.save(output_path, token_array)
    print(f"Successfully saved token IDs to {output_path} (dtype={token_array.dtype}, size={token_array.size}).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tokenize raw text data and save as NumPy array for Transformer training.")
    
    parser.add_argument('--raw_path', type=str, required=True, help="Path to the raw text file (e.g., TinyStories-train.txt).")
    parser.add_argument('--vocab_path', type=str, required=True, help="Path to the saved BPE vocabulary file (e.g., ts_vocab.pkl).")
    parser.add_argument('--merges_path', type=str, required=True, help="Path to the saved BPE merges file (e.g., ts_merges.txt).")
    parser.add_argument('--output_path', type=str, required=True, help="Output path for the tokenized NumPy array (.npy).")
    parser.add_argument('--special_tokens', type=str, default='<|endoftext|>', help="Comma-separated list of special tokens (defaults to EOS).")
    parser.add_argument('--dtype', type=str, default='uint16', help="NumPy dtype for token IDs (e.g., uint16).")
    
    args = parser.parse_args()
    
    special_tokens = [t.strip() for t in args.special_tokens.split(',')]
    
    tokenize_and_save(
        raw_text_path=args.raw_path,
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        output_path=args.output_path,
        special_tokens=special_tokens,
        dtype=args.dtype
    )

