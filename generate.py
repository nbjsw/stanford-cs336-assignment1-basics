import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import re
import pickle
from typing import Any, List, Optional, Iterable, Iterator

from utils import transformer_lm, dataloader, tokenizer 


# ----------------------------------------------------------------------
# 辅助函数：加载检查点和 Tokenizer 资产
# ----------------------------------------------------------------------

def load_checkpoint(filepath: str, model: torch.nn.Module) -> Any:
    """
    加载模型检查点。注意：解码只需要加载模型状态，不需要优化器状态。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}")
    
    # 强制将模型加载到 CPU，然后移动到目标设备
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    # print(f"Model state loaded successfully from {filepath}")
    
    return checkpoint.get('config', {})


def load_tokenizer_assets(vocab_path: str, merges_path: str, special_tokens: list[str]):
    """
    加载 Tokenizer 的词汇表和合并规则。
    """
    try:
        # 假设 tokenizer 模块有一个 load_merges_txt 函数
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
    应用 Softmax 温度缩放和 Top-p 采样，从 Logits 中采样下一个 Token ID。
    """
    # 1. 温度缩放 (Temperature Scaling)
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature
    
    # 2. 转换为概率分布
    probs = F.softmax(logits, dim=-1)

    # 3. Top-p 采样 (Nucleus Sampling)
    if top_p < 1.0:
        # 按概率降序排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # 计算累积概率和
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到第一个累积和超过 top_p 的位置
        mask = cumulative_probs > top_p
        
        # 将超过 top_p 的第一个元素设置为 False，以保留它
        # 这一步是为了确保保留达到 top_p 阈值的最小集合
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        
        # 将不保留的 Logits 设置为负无穷
        indices_to_remove = sorted_indices[mask]
        logits[indices_to_remove] = filter_value

        # 重新进行 Softmax，得到新的分布（只在 Top-p 集合内）
        probs = F.softmax(logits, dim=-1)

    # 4. 从最终概率分布中采样一个 Token
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
    context_length: int  # <-- 接受 context_length 参数
) -> str:
    """
    从模型中自回归地生成文本序列。
    """
    model.eval()
    
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids = prompt_ids.copy() # 记录生成的总序列
    
    eos_token_bytes = '<|endoftext|>'.encode('utf-8')
    eos_token_id = tokenizer_instance.vocab_to_id.get(eos_token_bytes, -1)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 1. 限制上下文长度 (如果输入序列超过模型支持的 context_length)
            seq_len = input_tensor.size(1)
            # 使用传入的 context_length 参数
            if seq_len > context_length: 
                input_tensor = input_tensor[:, -context_length:]
                
            # 2. 前向传播
            # 形状: (1, current_seq_len, vocab_size)
            logits = model(input_tensor)
            
            # 3. 获取最后一个 Token 的 Logits
            # 形状: (vocab_size)
            next_token_logits = logits[0, -1, :]
            
            # 4. 采样下一个 Token ID
            next_token_id_tensor = sample_next_token(
                next_token_logits, temperature, top_p
            )
            next_token_id = next_token_id_tensor.item()
            
            # 5. 检查终止条件
            if next_token_id == eos_token_id:
                break
                
            # 6. 更新序列
            generated_ids.append(next_token_id)
            
            # 将新 Token 添加到输入张量中，用于下一次迭代
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)

    # 7. 解码并返回结果
    return tokenizer_instance.decode(generated_ids)


def interactive_cli(args):
    # 1. 设备设置
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 2. 实例化模型 (需要匹配训练时的参数)
    model = transformer_lm.TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=10000 # 假设RoPE的theta参数是10000
    ).to(device)

    # 3. 加载检查点
    load_checkpoint(args.checkpoint_path, model)

    # 4. 加载 Tokenizer
    special_tokens = ['<|endoftext|>'] # 假设只有一个 EOS token
    tokenizer_instance = load_tokenizer_assets(args.vocab_path, args.merges_path, special_tokens)

    print("\n--- Model Loaded. Starting Interactive CLI ---")
    print(f"Model: L={args.num_layers}, H={args.num_heads}, D={args.d_model}")
    print(f"Generation Params: Temp={args.temperature}, Top-p={args.top_p}, Max Tokens={args.max_new_tokens}\n")
    
    # 5. 无限循环互动
    while True:
        try:
            prompt = input("输入提示 (或输入 'quit' 退出): ")
            
            if prompt.lower() in ('quit', 'exit'):
                print("程序退出。")
                break
            
            if not prompt.strip():
                continue

            # 编码 Prompt
            prompt_ids = tokenizer_instance.encode(prompt)
            
            if not prompt_ids:
                print("错误：无法编码提示文本，请尝试其他文本。")
                continue
                
            print("--- 模型生成中... ---")
            
            # 生成文本
            generated_text = generate_text(
                model,
                tokenizer_instance,
                prompt_ids,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                device,
                args.context_length  # <-- 直接传递 args.context_length
            )
            
            print("\n[AI 回复]:")
            print(generated_text)
            print("-" * 25)

        except Exception as e:
            print(f"\n[错误] 发生异常: {e}. 请重试或检查模型/Tokenizer设置。")
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactive console for a trained Transformer LM.")
    
    # --- 文件/路径参数 ---
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the trained model checkpoint (.pt).")
    parser.add_argument('--vocab_path', type=str, required=True, help="Path to the BPE vocabulary file (.pkl).")
    parser.add_argument('--merges_path', type=str, required=True, help="Path to the BPE merges file (.txt).")
    
    # --- 模型架构参数 (需要匹配训练时的参数) ---
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=1344)
    
    # --- 生成参数 ---
    parser.add_argument('--max_new_tokens', type=int, default=128, help="Maximum number of tokens to generate per response.")
    parser.add_argument('--temperature', type=float, default=0.8, help="Softmax temperature for sampling.")
    parser.add_argument('--top_p', type=float, default=0.9, help="Top-p (Nucleus) sampling threshold.")
    
    args = parser.parse_args()

    # 启动交互式 CLI
    interactive_cli(args)


