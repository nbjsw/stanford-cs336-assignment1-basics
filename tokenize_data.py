import numpy as np
import argparse
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
    """
    Streaming BPE tokenizer → streaming write to np.memmap (.npy).
    No full data load. Works for huge datasets (OpenWebText, C4).
    """
    print("--- Starting Tokenization (Streaming Mode) ---")
    tokenizer_instance = tokenizer.Tokenizer.from_files(
        vocab_path, merges_path, special_tokens
    )

    print(f"Tokenizer initialized. Vocab size = {len(tokenizer_instance.vocab)}")
    print(f"Preparing output file: {output_path}")
    f = open(output_path, "wb")

    # Reserve header space
    header_dict = {
        'descr': np.dtype(dtype).str,
        'fortran_order': False,
        'shape': (0,),  # placeholder, will rewrite later
    }
    write_array_header_1_0(f, header_dict)
    header_end = f.tell()  # remember where data starts

    print("Tokenizing and writing tokens (streaming)...")

    total_tokens = 0

    with open(raw_text_path, "r", encoding="utf-8") as fin:
        token_stream = tokenizer_instance.encode_iterable(fin)
        for token_id in tqdm(token_stream, unit=" token"):
            f.write(np.array(token_id, dtype=dtype).tobytes())
            total_tokens += 1

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

