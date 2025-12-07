import argparse
import numpy as np
import pickle
import re
from tqdm import tqdm
from numpy.lib.format import write_array_header_1_0
# 导入 Hugging Face 的 tokenizers 库组件
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from tokenizers.implementations import ByteLevelBPETokenizer

def load_merges_txt(filepath: str) -> list[tuple[bytes, bytes]]:
    """
    Load BPE merges from the custom .txt file format (using bytes repr).
    
    The file contains lines like: b"'" b's'
    We use eval() to safely interpret the bytes literal.
    """
    merges: list[tuple[bytes, bytes]] = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 使用 split() 分割字符串，而不是复杂的正则表达式。
            # 这依赖于合并规则之间只有一个空格。
            parts = line.split(' ')
            
            if len(parts) != 2:
                # 如果有多个空格或格式错误，它会在这里捕获
                raise ValueError(f"Merge line format error: Expected 2 parts, got {len(parts)} in line: {line}")
            
            p1_str, p2_str = parts
            
            try:
                # 使用 eval() 安全地将 bytes literal 字符串转换为实际的 bytes 对象
                p1_bytes = eval(p1_str) 
                p2_bytes = eval(p2_str)
            except Exception as e:
                raise RuntimeError(f"Failed to parse bytes literal using eval() for line: {line}. Error: {e}")

            if not isinstance(p1_bytes, bytes) or not isinstance(p2_bytes, bytes):
                raise TypeError(f"Parsed object is not bytes for line: {line}. Got {type(p1_bytes)}, {type(p2_bytes)}")
                
            merges.append((p1_bytes, p2_bytes))
            
    return merges

def get_hf_tokenizer(vocab_path: str, merges_path: str, special_tokens: list[str]) -> Tokenizer:
    """
    从自定义的 .pkl 和 .txt 文件加载 Byte-Level BPE 资产，并构建一个 HF Tokenizer 实例。
    """
    print("Loading custom BPE vocabulary and merges...")
    
    # 1. 加载 Vocab (pkl -> dict[int, bytes])
    with open(vocab_path, 'rb') as f:
        vocab_bytes: dict[int, bytes] = pickle.load(f)
        
    # 将 dict[int, bytes] 转换为 dict[str, int] (HF models.BPE 需要的格式)
    # Tokenizers 库的 BPE 模型需要 str 形式的 token
    vocab_str_to_id: dict[str, int] = {}
    for token_id, token_bytes in vocab_bytes.items():
        # 由于是 Byte-Level BPE，这里需要处理不可解码的字节，
        # 但 HF 的 Tokenizer BPE model 内部会处理这个。
        # 我们可以尝试使用 UTF-8 解码，如果失败则使用 bytes 的 repr 作为 token string
        try:
            token_str = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # 对于原始字节，使用它的 repr 字符串，这是自定义 BPE 的常见处理方式
            token_str = repr(token_bytes)
            
        vocab_str_to_id[token_str] = token_id

    # 2. 加载 Merges (txt -> list[tuple[bytes, bytes]])
    merges_bytes = load_merges_txt(merges_path)
    
    # 将 list[tuple[bytes, bytes]] 转换为 list[tuple[str, str]]
    merges_str: list[tuple[str, str]] = []
    for p1_bytes, p2_bytes in merges_bytes:
        # 使用与 Vocab 相同的逻辑来处理 bytes -> str 转换
        try:
            p1_str = p1_bytes.decode("utf-8")
        except UnicodeDecodeError:
            p1_str = repr(p1_bytes)
            
        try:
            p2_str = p2_bytes.decode("utf-8")
        except UnicodeDecodeError:
            p2_str = repr(p2_bytes)
            
        merges_str.append((p1_str, p2_str))


    # 3. 构建 Tokenizer
    
    # A. 构建 BPE 模型
    bpe_model = models.BPE(
        vocab=vocab_str_to_id,
        merges=merges_str,
        # Byte-Level BPE 的 UNK 默认为 None
    )

    # B. 构建完整的 Tokenizer 实例
    tokenizer = Tokenizer(bpe_model)

    # C. 设置 PreTokenizer (Byte-Level BPE 的关键)
    # ByteLevel pre-tokenizer 是与 ByteLevelBPETokenizer 训练时一致的关键
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # D. 添加特殊 Token
    # ByteLevelBPETokenizer 训练时会自动将特殊 token 视为整个 token，不需要复杂的添加
    tokenizer.add_special_tokens(special_tokens)

    # E. 设置 PostProcessor (可选，但推荐)
    # 假设第一个特殊 token 是 EOS (End-Of-Sequence)
    eos_token = special_tokens[0]
    eos_id = tokenizer.token_to_id(eos_token)
    
    if eos_id is not None:
        tokenizer.post_processor = processors.TemplateProcessing(
            single="$A",
            pair="$A $B",
            special_tokens=[(eos_token, eos_id)],
        )

    return tokenizer


# 将上一个回答中的 tokenize_and_save_hf 函数进行修改和整合
def tokenize_and_save_hf(
    raw_text_path: str,
    vocab_path: str,
    merges_path: str,
    output_path: str,
    special_tokens: list[str],
    dtype: str = 'uint16',
    batch_size: int = 1000 
):
    """
    使用 Hugging Face tokenizers 库进行流式 BPE 分词并写入 .npy。
    使用 HF 的优化来提速。
    """
    print("--- Starting Tokenization (HF Streaming Mode) ---")

    # 1. 初始化 HF Tokenizer (使用新实现的加载逻辑)
    tokenizer_instance = get_hf_tokenizer(vocab_path, merges_path, special_tokens)

    print(f"HF Tokenizer initialized. Vocab size = {tokenizer_instance.get_vocab_size()}")
    print(f"Preparing output file: {output_path}")
    
    # 2. 准备输出文件 (.npy header 逻辑不变)
    f = open(output_path, "wb")
    header_dict = {
        'descr': np.dtype(dtype).str,
        'fortran_order': False,
        'shape': (0,),
    }
    write_array_header_1_0(f, header_dict)
    header_end = f.tell() 
    
    print(f"Tokenizing and writing tokens (streaming with batch size {batch_size})...")

    total_tokens = 0
    buffer = bytearray() # 使用 bytearray 优化写入缓存
    
    # 3. 使用流式读取和 HF 批处理分词
    with open(raw_text_path, "r", encoding="utf-8") as fin:
        text_batch = []
        
        # 使用 tqdm 包装文件迭代器，显示进度
        for line in tqdm(fin, desc="Reading and Tokenizing"):
            # 添加 EOS token。原代码的分词器可能在内部处理，
            # 这里我们在行尾手动添加，以确保每个“文档”结束。
            # 这是一个关键的逻辑差异点，需要与原代码的 encode_iterable 行为一致。
            text_batch.append(line.strip() + special_tokens[0]) # 假设每个故事/行后接 EOS
            
            if len(text_batch) >= batch_size:
                # 使用 HF Tokenizer 的 encode_batch 提速
                encodings = tokenizer_instance.encode_batch(text_batch)
                
                # 将 batch 结果扁平化并写入缓存
                for encoding in encodings:
                    # 使用 np.fromiter 比 np.array(list).tobytes() 更快
                    token_ids = np.fromiter(encoding.ids, dtype=dtype)
                    buffer.extend(token_ids.tobytes())
                    total_tokens += len(token_ids)
                
                # 写入缓存并清空
                f.write(buffer)
                buffer = bytearray()
                text_batch = []

        # 处理剩余的 batch
        if text_batch:
            encodings = tokenizer_instance.encode_batch(text_batch)
            for encoding in encodings:
                token_ids = np.fromiter(encoding.ids, dtype=dtype)
                buffer.extend(token_ids.tobytes())
                total_tokens += len(token_ids)
            
            # 写入最终缓存
            f.write(buffer)

    # 4. 完善 .npy header 逻辑 (不变)
    print(f"Tokenization done. Total tokens = {total_tokens}")
    print("Finalizing .npy header...")
    f.seek(0)
    header_dict["shape"] = (total_tokens,)
    write_array_header_1_0(f, header_dict)
    f.close()
    print(f"Saved streaming tokenized data → {output_path}")
    print("Use np.load(..., mmap_mode='r') for training.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tokenize raw text data and save as NumPy array for Transformer training.")
    
    parser.add_argument('--raw_path', type=str, required=True, help="Path to the raw text file (e.g., TinyStories-train.txt).")
    parser.add_argument('--vocab_path', type=str, required=True, help="Path to the saved BPE vocabulary file (.pkl).")
    parser.add_argument('--merges_path', type=str, required=True, help="Path to the saved BPE merges file (.txt).")
    parser.add_argument('--output_path', type=str, required=True, help="Output path for the tokenized NumPy array (.npy).")
    parser.add_argument('--special_tokens', type=str, default='<|endoftext|>', help="Comma-separated list of special tokens (defaults to EOS).")
    parser.add_argument('--dtype', type=str, default='uint16', help="NumPy dtype for token IDs (e.g., uint16).")
    parser.add_argument('--batch_size', type=int, default=1000, help="Batch size for HF Tokenizer processing to speed up tokenization.")
    
    args = parser.parse_args()
    
    special_tokens = [t.strip() for t in args.special_tokens.split(',')]
    
    tokenize_and_save_hf(
        raw_text_path=args.raw_path,
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        output_path=args.output_path,
        special_tokens=special_tokens,
        dtype=args.dtype,
        batch_size=args.batch_size
    )
