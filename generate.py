import torch
import torch.nn.functional as F
import pickle
import argparse
import os
from typing import Any

from utils import transformer_lm, tokenizer

# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------

def load_tokenizer_assets(vocab_path: str, merges_path: str, special_tokens: list[str]):
    """
    加载 Tokenizer 的词汇表和合并规则。
    """
    try:
        merges = tokenizer.load_merges_txt(merges_path)
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        return tokenizer.Tokenizer(vocab, merges, special_tokens)
    except Exception as e:
        print(f"Error loading tokenizer assets: {e}")
        raise

def sample_next_token(
    logits: torch.Tensor, 
    temperature: float = 1.0, 
    top_p: float = 1.0,
    filter_value: float = -float('inf')
) -> torch.Tensor:
    """
    Softmax + top-p 采样
    """
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        indices_to_remove = sorted_indices[mask]
        logits[indices_to_remove] = filter_value
        probs = F.softmax(logits, dim=-1)

    next_token_id = torch.multinomial(probs, num_samples=1).squeeze(0)
    return next_token_id

def generate_text(
    model: torch.nn.Module,
    tokenizer_instance: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    context_length: int
) -> str:
    """
    自回归生成文本
    """
    model.eval()
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids = prompt_ids.copy()

    eos_token_bytes = '<|endoftext|>'.encode('utf-8')
    eos_token_id = tokenizer_instance.vocab_to_id.get(eos_token_bytes, -1)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_tensor.size(1) > context_length:
                input_tensor = input_tensor[:, -context_length:]

            logits = model(input_tensor)
            next_token_logits = logits[0, -1, :]
            next_token_id = sample_next_token(next_token_logits, temperature, top_p).item()

            if next_token_id == eos_token_id:
                break

            generated_ids.append(next_token_id)
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)

    return tokenizer_instance.decode(generated_ids)

# ----------------------------------------------------------------------
# CLI 主函数
# ----------------------------------------------------------------------

def interactive_cli(args):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. 加载 checkpoint
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model_config = checkpoint['config']
    model_state = checkpoint['model_state_dict']

    # 2. 实例化模型
    model = transformer_lm.TransformerLM(
        vocab_size=model_config['vocab_size'],
        context_length=model_config['context_length'],
        d_model=model_config['d_model'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        rope_theta=model_config['rope_theta'],
        device=device,
        dtype=eval(model_config['dtype']) if 'dtype' in model_config else torch.float32,
    )

    # 3. 加载权重
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    # 4. 加载 tokenizer
    special_tokens = ['<|endoftext|>']
    tokenizer_instance = load_tokenizer_assets(args.vocab_path, args.merges_path, special_tokens)

    print("\n--- Model Loaded. Starting Interactive CLI ---")
    print(f"Generation Params: Temp={args.temperature}, Top-p={args.top_p}, Max Tokens={args.max_new_tokens}\n")

    while True:
        try:
            prompt = input("输入提示 (或输入 'quit' 退出): ").strip()
            if prompt.lower() in ('quit', 'exit'):
                print("程序退出。")
                break
            if not prompt:
                continue

            prompt_ids = tokenizer_instance.encode(prompt)
            if not prompt_ids:
                print("错误：无法编码提示文本")
                continue

            print("--- 模型生成中... ---")
            generated_text = generate_text(
                model,
                tokenizer_instance,
                prompt_ids,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                device,
                model_config['context_length']
            )

            completion = generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text
            print("\n[AI 回复]:")
            print(completion)
            print("-" * 25)

        except Exception as e:
            print(f"[错误] {e}")
            continue

# ----------------------------------------------------------------------
# 程序入口
# ----------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactive console for a trained Transformer LM.")
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--merges_path', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.9)
    args = parser.parse_args()

    interactive_cli(args)
