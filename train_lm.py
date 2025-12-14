import torch
import numpy as np
import argparse
import time
import math
import os
import re
import queue
import threading
from einops import rearrange
from tqdm import tqdm
from utils import clip_grad, dataloader, transformer_lm, optimizer, lr, loss


class Prefetcher:
    """
    使用后台线程异步预加载数据的自定义加载器。
    """
    def __init__(self, data_source, batch_size, context_length, device, dtype, queue_size=5):
        self.data_source = data_source
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.dtype = dtype
        self.queue = queue.Queue(maxsize=queue_size)
        self.loader_thread = threading.Thread(target=self._data_loop, daemon=True)
        self.stop_event = threading.Event()

    def _data_loop(self):
        """后台线程持续加载数据并放入队列。"""
        # 注意: dataloader.data_loading 必须确保每次调用都返回下一个连续的批次，
        # 且将数据移动到 self.device 上。
        while not self.stop_event.is_set():
            try:
                # 假设 data_loading 已经处理了设备移动
                X, Y = dataloader.data_loading(
                    self.data_source, self.batch_size, self.context_length, self.device
                )
                self.queue.put((X, Y))
            except Exception as e:
                # 假设数据用尽或采样出错
                if not self.stop_event.is_set():
                    # print(f"Prefetcher data loading error: {e}. Retrying.")
                    time.sleep(0.1)

    def start(self):
        """启动后台加载线程。"""
        self.loader_thread.start()

    def next(self):
        """主线程获取下一个预加载的批次数据。移除 .clone() 以优化性能。"""
        # 假设数据在放入队列时已位于正确的设备和 dtype
        X, Y = self.queue.get(timeout=30)
        return X, Y

    def stop(self):
        """停止后台加载线程。"""
        self.stop_event.set()
        if self.loader_thread.is_alive():
             # 清空队列，防止 join 阻塞
             while not self.queue.empty():
                 self.queue.get()
             self.loader_thread.join(timeout=1)


def find_latest_checkpoint(checkpoint_dir: str) -> tuple[int, str | None]:
    # ... (与您的代码相同，用于查找最新的检查点) ...
    if not os.path.isdir(checkpoint_dir):
        return 0, None

    pattern = re.compile(r"checkpoint_step_(\d+)\.pt$")
    max_iteration = 0
    latest_checkpoint_path = None

    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if match:
            current_iteration = int(match.group(1))
            if current_iteration > max_iteration:
                max_iteration = current_iteration
                latest_checkpoint_path = os.path.join(checkpoint_dir, filename)

    return max_iteration, latest_checkpoint_path


def calculate_perplexity(loss: float) -> float:
    """Calculates perplexity (PPL) from the average cross-entropy loss."""
    return math.exp(loss)


def evaluate(model, data, context_length, batch_size, device, dtype) -> tuple[float, float]:
    """Runs model in evaluation mode on validation data."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    max_eval_batches = 100 
    
    with torch.no_grad():
        for _ in range(max_eval_batches):
            try:
                X, Y = dataloader.data_loading(data, batch_size, context_length, device)
            except Exception:
                break 
            
            with torch.amp.autocast('cuda', dtype=dtype):
                logits = model(X)
                criterion = loss.CrossEntropyLoss()
                # 展平 Logits 和 Targets
                flat_logits = rearrange(logits, 'b l v -> (b l) v')
                flat_targets = rearrange(Y, 'b l -> (b l)')
                cur_loss = criterion(flat_logits, flat_targets)
            
                total_loss += cur_loss.item()
                num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0

    avg_loss = total_loss / num_batches
    avg_ppl = calculate_perplexity(avg_loss)
    return avg_loss, avg_ppl


def save_checkpoint(
    model: torch.nn.Module,
    optimizer_ins: torch.optim.Optimizer,
    iteration: int,
    full_checkpoint_path: str,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: int,
):
    # 假设 model 是被 torch.compile(raw_model) 包装后的对象
    if hasattr(model, '_orig_mod'):
        model_to_save = model._orig_mod.state_dict()
    elif hasattr(model, 'module'):
        model_to_save = model.module.state_dict()
    else:
        model_to_save = model.state_dict()

    # 构造要保存的状态字典
    checkpoint = {
        'model_state_dict': model_to_save,
        'optimizer_state_dict': optimizer_ins.state_dict(),
        'iteration': iteration,
        'config': {
            'vocab_size': vocab_size,
            'context_length': context_length,
            'd_model': d_model,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'd_ff': d_ff,
            'rope_theta': rope_theta,
            'dtype': str(next(model.parameters()).dtype),
        }
    }
    # 确保目录存在
    os.makedirs(os.path.dirname(full_checkpoint_path) or '.', exist_ok=True)
    # 使用 torch.save 进行保存
    torch.save(checkpoint, full_checkpoint_path)
    print(f"Checkpoint saved to {full_checkpoint_path} (Resuming iter: {iteration})")


def train(args):
    # 1. 设备设置 (Device Setup)
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        device = args.device
    
    # 使用 bfloat16 进行计算，但将 token ID 保留为 long 或 int
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32
        
    print(f"Using device: {device}, compute_dtype: {compute_dtype}")

    # 2. 数据加载 (Data Loading)
    print(f"Loading training data from {args.train_data_path}")
    train_data = np.memmap(args.train_data_path, dtype=args.dtype, mode='r')
    print(f"Loading validation data from {args.val_data_path}")
    val_data = np.memmap(args.val_data_path, dtype=args.dtype, mode='r')
    
    # 3. 模型实例化 (Model Instantiation)
    model = transformer_lm.TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=10000,
        device=device,
        dtype=compute_dtype
    )
    
    print("Compiling model using torch.compile...")
    model = torch.compile(model, mode='default')
    print("Model compilation finished.")
    
    # 4. 优化器设置 (Optimizer Setup)
    optimizer_instance = optimizer.MyAdamW(
        model.parameters(),
        lr=args.max_lr, 
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    
    # 5. Loss
    criterion = loss.CrossEntropyLoss()
    
    # 5. 检查点和恢复 (Checkpointing and Resumption)
    max_iteration, latest_ckpt_path = find_latest_checkpoint(args.checkpoint_path)
    start_iteration = max_iteration
    # ... (检查点加载逻辑，与您的代码相同) ...
    if start_iteration >= args.max_iters:
        print(f"Training already complete! Max iterations ({args.max_iters}) reached at step {start_iteration}.")
        return
    if latest_ckpt_path:
        print(f"Resuming training. Attempting to load checkpoint from {latest_ckpt_path}...")
        try:
            # 假设你的 dataloader.load_checkpoint 是正确的加载函数
            dataloader.load_checkpoint(latest_ckpt_path, model, optimizer_instance)
            print(f"Checkpoint loaded successfully. Resuming from iteration {start_iteration}.")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {latest_ckpt_path} ({e}). Starting new run from iteration 0.")
            start_iteration = 0
    else:
        print("No checkpoint found. Starting new run from iteration 0.")


    # 6. 训练循环 (Training Loop)
    model.train()
    pbar = tqdm(range(start_iteration, args.max_iters), initial=start_iteration, total=args.max_iters, desc="Training")

    # --- 启动异步数据预加载器 ---
    print("Starting asynchronous data prefetcher...")
    prefetcher = Prefetcher(
        train_data, 
        args.batch_size, 
        args.context_length, 
        device,
        compute_dtype,
        queue_size=10
    )
    prefetcher.start()
    
    # 预取 N 个批次用于第一次优化器更新
    prefetched_batches = []
    for _ in range(args.accumulation_steps):
        prefetched_batches.append(prefetcher.next())

    # --- 关键：主训练循环 ---
    try:
        for iteration in pbar:
            # 获取学习率并更新优化器
            learing_rate = lr.calculate_cosine_annealing_lr(
                iteration, args.max_lr, args.min_lr, args.warmup_steps, args.decay_steps
            )
            for param_group in optimizer_instance.param_groups:
                param_group['lr'] = learing_rate
                
            # 1. 梯度归零 (Zero Gradients)
            optimizer_instance.zero_grad()
            
            # --- 核心：梯度累加循环 ---
            current_train_loss = 0.0
            
            # 计算本迭代的随机 RoPE 偏移
            # Llama/Mistral 风格：在训练中随机化 RoPE 绝对位置
            B, L = prefetched_batches[0][0].shape # 从第一个预取批次获取 B 和 L
            max_offset = args.context_length # 设置一个偏移上限

            # 随机选择一个偏移量 S (例如，在 0 到 L 之间)
            # 这样模型就不知道自己是不是在序列的开头
            random_offset = torch.randint(0, max_offset + 1, (1,)).item()

            # 创建偏移后的位置张量: [S, S+1, ..., S+L-1]
            positions = torch.arange(L, device=device, dtype=torch.long) + random_offset
            # 扩展到 (B, L)
            token_positions = positions.unsqueeze(0).expand(B, L)

            # 确保 token_positions 的 dtype 匹配 Attention/RoPE 的要求 (通常是 torch.long)
            token_positions = token_positions.to(device)

            for micro_step in range(args.accumulation_steps):
                # 从预取列表中获取当前微步的数据 (连续小批次)
                X, Y = prefetched_batches[micro_step]
                X = X.clone()
                Y = Y.clone()
                torch.compiler.cudagraph_mark_step_begin()
                with torch.amp.autocast('cuda', dtype=compute_dtype):
                    # 2. 前向传播 (Forward Pass)
                    logits = model(X, token_positions=token_positions)

                    # 展平 Logits 和 Targets Y
                    flat_logits = rearrange(logits, 'b l v -> (b l) v')
                    flat_targets = rearrange(Y, 'b l -> (b l)')

                    # 3. 计算损失并缩放
                    # 损失除以累积步数，保证反向传播时梯度幅度正确
                    cur_loss = criterion(flat_logits, flat_targets) / args.accumulation_steps
                    current_train_loss += cur_loss.item()
                
                # 4. 反向传播 (Backward Pass)
                # 梯度被累加到 .grad 属性中
                cur_loss.backward()

            # --- 梯度更新步骤 (Optimizer Update) ---
            
            # 5. 梯度裁剪 (Gradient Clipping)
            clip_grad.gradient_clipping(model.parameters(), args.grad_clip_norm)
            
            # 6. 优化器步骤 (Optimizer Step)
            torch.compiler.cudagraph_mark_step_begin()
            optimizer_instance.step()

            # 7. 启动下一轮的 N 个批次预取
            # 在 GPU 计算完成后，立即在后台启动下一轮 N 个批次的数据加载
            prefetched_batches = []
            for _ in range(args.accumulation_steps):
                prefetched_batches.append(prefetcher.next())
                
            # 8. 日志记录和评估 (Logging and Evaluation)
            if iteration % args.log_interval == 0:
                # 记录的是累积后的平均损失
                train_loss = current_train_loss
                train_ppl = calculate_perplexity(train_loss)
                pbar.set_postfix({'loss': f'{train_loss:.4f}', 'ppl': f'{train_ppl:.1f}', 'lr': f'{learing_rate:.2e}'})

            if iteration % args.eval_interval == 0 and iteration > start_iteration:
                print("\n--- Running Validation ---")
                val_loss, val_ppl = evaluate(
                    model, val_data, args.context_length, args.batch_size, device, compute_dtype
                )
                print(f"[EVAL] Iter {iteration}: Val Loss={val_loss:.4f}, Val PPL={val_ppl:.1f}")
                model.train()
                print("--------------------------")
            
            # 9. 检查点 (Checkpointing)
            if iteration > 0 and iteration % args.save_interval == 0 and iteration > start_iteration:
                checkpoint_filename = f"checkpoint_step_{iteration}.pt"
                full_checkpoint_path = os.path.join(args.checkpoint_path, checkpoint_filename) 
                save_checkpoint(model, optimizer_instance, iteration + 1, full_checkpoint_path,
                                             args.vocab_size, args.context_length, args.d_model, args.num_layers,
                                             args.num_heads, args.d_ff, 10000)

    except Exception as e:
        print(f"\nTraining loop error: {e}")
    finally:
        prefetcher.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model from Scratch (CS336 A1).")

    # --- 数据/路径参数 ---
    parser.add_argument('--train_data_path', type=str, required=True, help="Path to the tokenized training data (NumPy array).")
    parser.add_argument('--val_data_path', type=str, required=True, help="Path to the tokenized validation data (NumPy array).")
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/run1', help="Path to save/load model checkpoints.")
    parser.add_argument('--dtype', type=str, default='uint16', choices=['uint16', 'int32'], help="Data type of the tokenized files (e.g., uint16 for TinyStories).")

    # --- 模型架构参数 (假设已在 utils/transformer_lm.py 中加入了 dropout) ---
    parser.add_argument('--vocab_size', type=int, default=32768, help="Vocabulary size.")
    parser.add_argument('--context_length', type=int, default=256, help="Maximum sequence length.")
    parser.add_argument('--d_model', type=int, default=1280, help="Hidden dimension (d_model).")
    parser.add_argument('--num_layers', type=int, default=12, help="Number of Transformer blocks.")
    parser.add_argument('--num_heads', type=int, default=16, help="Number of attention heads.")
    parser.add_argument('--d_ff', type=int, default=3456, help="Feed-forward inner dimension (d_ff).")

    # --- 训练参数 (调整以解决过拟合和低学习率) ---
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'], help="Device to use for training (auto detects CUDA/MPS).")
    parser.add_argument('--max_iters', type=int, default=20000, help="Total number of training iterations.")
    parser.add_argument('--batch_size', type=int, default=32, help="Actual batch size (显存限制).")
    parser.add_argument('--accumulation_steps', type=int, default=16, help="Gradient accumulation steps (等效 B=512).")

    # --- 优化器/调度器参数 (调整以解决 PPL 飙升问题) ---
    parser.add_argument('--max_lr', type=float, default=5e-5, help="Maximum learning rate.")
    parser.add_argument('--min_lr', type=float, default=5e-6, help="Minimum learning rate.")
    parser.add_argument('--warmup_steps', type=int, default=2000, help="Number of steps for linear LR warm-up (缩短 Warmup).")
    parser.add_argument('--decay_steps', type=int, default=18000, help="Number of steps for cosine annealing decay.")
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help="Maximum L2 norm for gradient clipping.")

    # AdamW specific (调整以增强稳定性)
    parser.add_argument('--beta1', type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument('--beta2', type=float, default=0.999, help="AdamW beta2.")
    parser.add_argument('--eps', type=float, default=1e-8, help="AdamW epsilon for numerical stability.")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="AdamW weight decay (降低正则化强度).")

    # --- 日志/保存间隔 ---
    parser.add_argument('--log_interval', type=int, default=10, help="Interval for logging training loss to console/tqdm.")
    parser.add_argument('--eval_interval', type=int, default=200, help="Interval for running validation evaluation (略微延长间隔).")
    parser.add_argument('--save_interval', type=int, default=1000, help="Interval for saving checkpoints.")

    args = parser.parse_args()
    train(args)
