import numpy as np
import argparse
import array
from tqdm import tqdm
from utils import tokenizer
from numpy.lib.format import write_array_header_1_0


def tokenize_and_save(
    raw_text_path: str,
    vocab_path: str,
    merges_path: str,
    output_path: str,
    special_tokens: list[str],
    dtype: str = 'uint16'
):
    print("--- Starting Tokenization (Batch Extension Mode) ---")
    tokenizer_instance = tokenizer.Tokenizer.from_files(
        vocab_path, merges_path, special_tokens
    )

    # 缓冲区大小：500MB
    BUFFER_SIZE = 250_000_000 
    array_type_code = 'H' if dtype == 'uint16' else 'I'

    print(f"Buffer size: {BUFFER_SIZE:,} tokens. Writing to {output_path}")

    f = open(output_path, "wb")
    
    header_dict = {
        'descr': np.dtype(dtype).str,
        'fortran_order': False,
        'shape': (0,),  
    }
    write_array_header_1_0(f, header_dict)

    print("Tokenizing...")

    total_tokens = 0
    buffer = array.array(array_type_code)

    pbar = tqdm(unit=" tokens", mininterval=1.0)

    with open(raw_text_path, "r", encoding="utf-8") as fin:
        # 批量读取多行，减少 Python I/O 开销
        BATCH_LINES = 1000 
        lines_buffer = []

        for line in fin:
            lines_buffer.append(line)
            
            if len(lines_buffer) >= BATCH_LINES:
                # 1. 拼接文本 (Python 字符串拼接很快)
                text_chunk = "".join(lines_buffer)
                
                # 2. Rust 一次性处理一大块文本，返回一个大 list[int]
                tokens_list = tokenizer_instance.encode(text_chunk)
                
                # 3. 【核心提速点】直接 extend，完全避开 Python 循环
                # array.extend 在 C 层面执行，速度极快
                buffer.extend(tokens_list)
                
                # 更新统计
                total_tokens += len(tokens_list)
                pbar.update(len(tokens_list))
                lines_buffer.clear()

                # 4. 检查缓冲区是否写盘
                if len(buffer) >= BUFFER_SIZE:
                    buffer.tofile(f)
                    buffer = array.array(array_type_code) # 重置

        # 处理剩余的行
        if lines_buffer:
            text_chunk = "".join(lines_buffer)
            tokens_list = tokenizer_instance.encode(text_chunk)
            buffer.extend(tokens_list)
            total_tokens += len(tokens_list)
            pbar.update(len(lines_buffer))

        # 处理剩余的 buffer
        if len(buffer) > 0:
            buffer.tofile(f)
    
    pbar.close()
    f.seek(0)
    header_dict["shape"] = (total_tokens,)
    write_array_header_1_0(f, header_dict)
    f.close()
    
    print(f"\nDone. Total tokens: {total_tokens:,}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="High-Performance Tokenizer")
    parser.add_argument('--raw_path', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--merges_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--special_tokens', type=str, default='<|endoftext|>')
    parser.add_argument('--dtype', type=str, default='uint16', choices=['uint16', 'uint32'])
    
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
