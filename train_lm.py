import torch
import numpy as np
import argparse
import time
import math
import os
import re
from einops import rearrange
from tqdm import tqdm
from utils import clip_grad, dataloader, transformer_lm, optimizer, lr, loss

class Prefetcher:
    """
    使用后台线程异步预加载数据的自定义加载器。
    这是解决 GPU I/O 瓶颈，实现计算与数据加载重叠的关键。
    """
    def __init__(self, data_source, batch_size, context_length, device, queue_size=5):
        self.data_source = data_source
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        # 使用 Queue 来在线程间安全地传递数据
        self.queue = queue.Queue(maxsize=queue_size)
        # 启动一个后台线程来执行数据加载
        self.loader_thread = threading.Thread(target=self._data_loop, daemon=True)
        self.stop_event = threading.Event()

    def _data_loop(self):
        """后台线程持续加载数据并放入队列。"""
        # 循环，直到接收到停止事件
        while not self.stop_event.is_set():
            try:
                X, Y = dataloader.data_loading(
                    self.data_source, self.batch_size, self.context_length, self.device
                )
                # 将加载好的数据放入队列，如果队列满了，线程会在这里阻塞
                self.queue.put((X, Y))
            except Exception as e:
                # 假设数据用尽或采样出错，等待一会儿再重试或退出
                # 在实际训练中，如果数据用尽，应该重置采样器或退出
                if not self.stop_event.is_set():
                    # print(f"Prefetcher data loading error: {e}. Retrying.")
                    time.sleep(0.1)

    def start(self):
        """启动后台加载线程。"""
        self.loader_thread.start()

    def next(self):
        """主线程获取下一个预加载的批次数据。"""
        # 阻塞等待直到队列中有数据
        return self.queue.get(timeout=30) # 设置超时时间，防止死锁

    def stop(self):
        """停止后台加载线程。"""
        self.stop_event.set()
        # 确保线程能被清理，join等待线程结束
        if self.loader_thread.is_alive():
             self.loader_thread.join(timeout=1)


def find_latest_checkpoint(checkpoint_dir: str) -> tuple[int, str | None]:
    """
          扫描检查点目录，找出最大迭代步数的检查点文件。
    
    Args:
        checkpoint_dir: 检查点保存的目录路径。
        
    Returns:
        (max_iteration, latest_checkpoint_path): 最大步数和对应的文件路径，
                                                如果找不到任何文件，则返回 (0, None)。
    """
    if not os.path.isdir(checkpoint_dir):
        return 0, None

    # 正则表达式用于匹配文件名中的步数，例如 'checkpoint_step_12345.pt'
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


def evaluate(model, data, context_length, batch_size, device) -> tuple[float, float]:
    """Runs model in evaluation mode on validation data."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # 简化评估：只评估 100 个批次以节省时间
    max_eval_batches = 100 
    
    with torch.no_grad():
        for _ in range(max_eval_batches):
            try:
                # dataloader.data_loading 必须能够从 np.memmap 中高效地采样
                X, Y = dataloader.data_loading(data, batch_size, context_length, device)
            except Exception:
                # 假设数据用尽或采样出错
                break 

            logits = model(X)
            # 损失模块通常有一个 forward 方法，这里直接调用即可
            criterion = loss.CrossEntropyLoss() # 在 evaluate 中重新实例化损失模块
            cur_loss = criterion(logits, Y)
            
            total_loss += cur_loss.item()
            num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0

    avg_loss = total_loss / num_batches
    avg_ppl = calculate_perplexity(avg_loss)
    return avg_loss, avg_ppl


def save_checkpoint(model: torch.nn.Module, optimizer_ins: torch.optim.Optimizer, iteration: int, full_checkpoint_path: str):
    """
    Saves the model state, optimizer state, and current iteration number to a file.
    
    Args:
        model: The torch.nn.Module instance to save.
        optimizer_ins: The torch.optim.Optimizer instance to save.
        iteration: The next iteration number to resume from (e.g., current iteration + 1).
        full_checkpoint_path: The full path to the checkpoint file (e.g., 'checkpoints/run1/checkpoint_step_5000.pt').
    """
    # 假设 model 是被 torch.compile(raw_model) 包装后的对象
    if hasattr(model, '_orig_mod'):
        # 这是 torch.compile 的解包方法
        model_to_save = model._orig_mod.state_dict()
    elif hasattr(model, 'module'):
        # 这是 DDP 的解包方法
        model_to_save = model.module.state_dict()
    else:
        model_to_save = model.state_dict()

    # 构造要保存的状态字典
    checkpoint = {
        'model_state_dict': model_to_save,
        'optimizer_state_dict': optimizer_ins.state_dict(),
        'iteration': iteration,
        # 可以选择性地保存模型配置 (args)
        # 'config': args_dict 
    }
    # 使用 torch.save 进行保存
    torch.save(checkpoint, full_checkpoint_path)
    print(f"Checkpoint saved to {full_checkpoint_path} (Resuming iter: {iteration})")


def train(args):
    # 1. 设备设置 (Device Setup)
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    # 2. 数据加载 (Data Loading) - 内存高效
    # 使用 np.memmap 进行内存高效的大文件加载
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
        rope_theta=10000
    ).to(device)
    
    # 可选：使用 torch.compile 优化性 (保留)
    print("Compiling model using torch.compile...")
    model = torch.compile(model, mode='reduce-overhead')
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

    # A. 检查是否训练完成
    if start_iteration >= args.max_iters:
        print(f"Training already complete! Max iterations ({args.max_iters}) reached at step {start_iteration}.")
        return
    # B. 加载最新的检查点
    if latest_ckpt_path:
        print(f"Resuming training. Attempting to load checkpoint from {latest_ckpt_path}...")
        try:
            # 假设你的 load_checkpoint 签名是 (filepath, model, optimizer)
            # 注意: 这里的 load_checkpoint 应该返回保存时的 'iteration + 1'，但我们直接使用 find_latest_checkpoint 得到的 max_iteration
            # 为了严谨，如果 load_checkpoint 失败，我们从 0 开始
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
        queue_size=5 # 队列大小，可调整
    )
    prefetcher.start()

    # 预取第一个批次，确保循环启动时就有数据
    # 这是 Iteration 0 的 X, Y
    X, Y = prefetcher.next() 
    
    # 预取下一个批次，以便在 Iteration 0 的 GPU 计算时加载 Iteration 1 的数据
    # 这是实现计算/I/O 重叠的关键
    next_X, next_Y = prefetcher.next() 
    # -----------------------------
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
            
            # 2. 前向传播 (Forward Pass)
            logits = model(X)

            # 步骤一：展平 Logits
            flat_logits = rearrange(logits, 'b l v -> (b l) v')

            # 步骤二：展平 Targets Y
            flat_targets = rearrange(Y, 'b l -> (b l)')

            # 3. 计算损失 (Calculate Loss)
            cur_loss = criterion(flat_logits, flat_targets)

            # 4. 反向传播 (Backward Pass)
            cur_loss.backward()

            # 5. 异步数据加载 (Prefetching)
            # 在这里调用 next()，Prefetcher 线程会阻塞并等待 GPU 完成计算，
            # 确保在 GPU 空闲时 CPU 立即将数据放入队列。
            # 这就是计算和 I/O 的重叠！
            new_next_X, new_next_Y = prefetcher.next()

            # 6. 梯度裁剪 (Gradient Clipping)
            clip_grad.gradient_clipping(model.parameters(), args.grad_clip_norm)
            
            # 7. 优化器步骤 (Optimizer Step)
            # GPU compute intensive
            optimizer_instance.step()

            # 8. 交换数据：为下一轮迭代准备预加载好的数据
            # Iteration N 的 X, Y -> Iteration N+1 的 X, Y
            X, Y = next_X, next_Y
            # 更新预加载的下一批数据
            next_X, next_Y = new_next_X, new_next_Y

        # 8. 日志记录和评估 (Logging and Evaluation)
        if iteration % args.log_interval == 0:
            train_loss = cur_loss.item()
            train_ppl = calculate_perplexity(train_loss)
            pbar.set_postfix({'loss': f'{train_loss:.4f}', 'ppl': f'{train_ppl:.1f}', 'lr': f'{learing_rate:.2e}'})
            
            # 实际中可以在这里集成 wandb.log({'train/loss': train_loss, 'learing_rate': lr}, step=iteration)

        if iteration % args.eval_interval == 0 and iteration > start_iteration:
            print("\n--- Running Validation ---")
            val_loss, val_ppl = evaluate(
                model, val_data, args.context_length, args.batch_size, device
            )
            print(f"[EVAL] Iter {iteration}: Val Loss={val_loss:.4f}, Val PPL={val_ppl:.1f}")
            # 实际中可以在这里集成 wandb.log({'val/loss': val_loss, 'val/ppl': val_ppl}, step=iteration)
            model.train() # 切换回训练模式
            print("--------------------------")

        # 9. 检查点 (Checkpointing)
        if iteration > 0 and iteration % args.save_interval == 0 and iteration > start_iteration:
            # 构造新的文件路径，包含步数
            # 示例: 'checkpoints/tinystories_base_run/checkpoint_step_5000.pt'
            checkpoint_filename = f"checkpoint_step_{iteration}.pt"
    
            # 假设 args.checkpoint_path 是目录路径，使用 os.path.join 拼接完整路径
            full_checkpoint_path = os.path.join(args.checkpoint_path, checkpoint_filename) 

            # 确保目录存在 (需要在训练开始前导入 os 并在训练启动时执行)
            # os.makedirs(args.checkpoint_path, exist_ok=True)

            save_checkpoint(model, optimizer_instance, iteration + 1, full_checkpoint_path)

    except Exception as e:
        print(f"\nTraining loop error: {e}")
    finally:
        # 训练结束或遇到异常时，确保停止预加载器
        prefetcher.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model from Scratch (CS336 A1).")

    # --- 数据/路径参数 (Data/Path Arguments) ---
    parser.add_argument('--train_data_path', type=str, required=True, help="Path to the tokenized training data (NumPy array).")
    parser.add_argument('--val_data_path', type=str, required=True, help="Path to the tokenized validation data (NumPy array).")
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pt', help="Path to save/load model checkpoints.")
    parser.add_argument('--dtype', type=str, default='uint16', choices=['uint16', 'int32'], help="Data type of the tokenized files (e.g., uint16 for TinyStories).")

    # --- 模型架构参数 (Model Architecture Arguments) ---
    parser.add_argument('--vocab_size', type=int, default=32768, help="Vocabulary size.")
    parser.add_argument('--context_length', type=int, default=256, help="Maximum sequence length.")
    parser.add_argument('--d_model', type=int, default=1280, help="Hidden dimension (d_model).")
    parser.add_argument('--num_layers', type=int, default=12, help="Number of Transformer blocks.")
    parser.add_argument('--num_heads', type=int, default=16, help="Number of attention heads.")
    parser.add_argument('--d_ff', type=int, default=3456, help="Feed-forward inner dimension (d_ff).")

    # --- 训练参数 (Training Arguments) ---
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'], help="Device to use for training (auto detects CUDA/MPS).")
    parser.add_argument('--max_iters', type=int, default=5000, help="Total number of training iterations.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")

    # --- 优化器/调度器参数 (Optimizer/Scheduler Arguments) ---
    parser.add_argument('--max_lr', type=float, default=5e-4, help="Maximum learning rate.")
    parser.add_argument('--min_lr', type=float, default=1e-4, help="Minimum learning rate.")
    parser.add_argument('--warmup_steps', type=int, default=50, help="Number of steps for linear LR warm-up.")
    parser.add_argument('--decay_steps', type=int, default=4500, help="Number of steps for cosine annealing decay.")
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help="Maximum L2 norm for gradient clipping.")

    # AdamW specific
    parser.add_argument('--beta1', type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument('--beta2', type=float, default=0.999, help="AdamW beta2 (commonly 0.95 for LLMs).")
    parser.add_argument('--eps', type=float, default=1e-8, help="AdamW epsilon for numerical stability.")
    parser.add_argument('--weight_decay', type=float, default=0.1, help="AdamW weight decay.")

    # --- 日志/保存间隔 (Logging/Save Intervals) ---
    parser.add_argument('--log_interval', type=int, default=10, help="Interval for logging training loss to console/tqdm.")
    parser.add_argument('--eval_interval', type=int, default=100, help="Interval for running validation evaluation.")
    parser.add_argument('--save_interval', type=int, default=200, help="Interval for saving checkpoints.")

    args = parser.parse_args()
    train(args)

