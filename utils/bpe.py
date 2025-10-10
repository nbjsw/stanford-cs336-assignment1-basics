import regex as re
from collections import defaultdict
# 替换 process_map 为标准的 map，避免多进程环境带来的挂起问题
# from tqdm.contrib.concurrent import process_map
from typing import Dict, Tuple, List, Any

# ----------------------------------------------------------------------
# 核心 BPE 训练函数
# ----------------------------------------------------------------------
# 与原代码保持一致
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def read_text(input_path: str) -> str:
    """读取文件内容."""
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def split_by_special(text: str, special_tokens: List[str], drop_special: bool = True) -> List[str]:
    """与原代码保持一致: 使用特殊Token分割文本."""
    if not special_tokens:
        return [text]

    special_tokens = sorted(special_tokens, key=len, reverse=True)

    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if not drop_special: pattern = f"({pattern})"

    pattern = re.compile(pattern)
    chunks = pattern.split(text)
    return [c for c in chunks if c]


def word2bytes(word: str) -> Tuple[bytes, ...]:
    """将词字符串转换为字节元组 (与原代码完全一致)."""
    a = list(word.encode('utf-8'))
    return tuple(bytes([i]) for i in a)


def count_word(text: str) -> Dict[Tuple[bytes, ...], int]:
    """分割文本并统计词汇字节频率 (与原代码完全一致)."""
    word_cnt = defaultdict(int)
    for m in PAT.finditer(text):
        word = m.group(0)
        word_bytes = word2bytes(word)
        if len(word_bytes) >= 2:
            word_cnt[word_bytes] += 1
    return word_cnt


def merge_dicts(dicts: List[Dict[Any, int]]) -> Dict[Any, int]:
    """合并词频字典 (与原代码完全一致)."""
    merged = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            merged[k] += v
    return merged


def count_pair(word_cnt: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, bytes], int]:
    """统计所有相邻对的频率 (与原代码完全一致)."""
    pair_cnt = defaultdict(int)
    for word_bytes, cnt in word_cnt.items():
        for i in range(len(word_bytes) - 1):
             pair_cnt[(word_bytes[i], word_bytes[i+1])] += cnt
    return pair_cnt


def get_max_pair(pair_cnt: Dict[Tuple[bytes, bytes], int]) -> Tuple[bytes, bytes]:
    """查找最大频率的对，使用字典序作为tie-breaker (与原代码完全一致)."""
    max_pair, _ = max(pair_cnt.items(), key=lambda x: (x[1], x[0]))
    return max_pair


def get_basic_vocab(special_tokens: List[str]) -> Dict[int, bytes]:
    """构建基础词汇表 (与原代码完全一致)."""
    vocab = {token: bytes([token]) for token in range(256)}
    for i, token in enumerate(special_tokens):
        token_id = 256 + i
        vocab[token_id] = token.encode("utf-8")
    return vocab


def apply_merge(word_bytes: Tuple[bytes, ...], merge: Tuple[bytes, bytes]) -> Tuple[bytes, ...]:
    """将一个词汇中的所有匹配对进行合并 (与原代码完全一致)."""
    merged = merge[0] + merge[1]
    i = 0
    new_word_bytes: List[bytes] = []
    while i < len(word_bytes):
        # 检查是否匹配
        if i < len(word_bytes) - 1 and word_bytes[i] == merge[0] and word_bytes[i + 1] == merge[1]:
            new_word_bytes.append(merged)
            i += 2
        else:
            new_word_bytes.append(word_bytes[i])
            i += 1
    return tuple(new_word_bytes)


def update_cnt_optimized(
    word_cnt: Dict[Tuple[bytes, ...], int],
    pair_cnt: Dict[Tuple[bytes, bytes], int],
    merge_pair: Tuple[bytes, bytes]
) -> Tuple[Dict[Tuple[bytes, ...], int], Dict[Tuple[bytes, bytes], int]]:
    """
    **性能优化版本 (与 V1 相同)**: 仅处理包含 `merge_pair` 的词汇。
    """
    
    new_word_cnt = defaultdict(int)
    new_pair_cnt = defaultdict(int, pair_cnt)

    words_to_update: Dict[Tuple[bytes, ...], int] = {}
    
    # ----------------------------------------------------------------
    # 第一步: 筛选出包含 merge_pair 的词汇并复制未改变的词汇
    # ----------------------------------------------------------------
    for word_bytes, cnt in word_cnt.items():
        has_merge = False
        if len(word_bytes) >= 2:
            # 快速检查是否存在 merge_pair
            for i in range(len(word_bytes) - 1):
                if (word_bytes[i], word_bytes[i+1]) == merge_pair:
                    has_merge = True
                    break
        
        if has_merge:
            words_to_update[word_bytes] = cnt
        else:
            # 直接复制到新的词频字典
            new_word_cnt[word_bytes] += cnt

    # ----------------------------------------------------------------
    # 第二步: 更新包含 merge_pair 的词汇和相应的 pair_cnt
    # ----------------------------------------------------------------
    for word_bytes, cnt in words_to_update.items():
        # 1. 查找所有旧对
        old_pairs = list(zip(word_bytes[:-1], word_bytes[1:]))

        # 2. 从 pair_cnt 中减去旧对的计数 (与原代码一致)
        for pair in old_pairs:
            new_pair_cnt[pair] -= cnt
            if new_pair_cnt[pair] == 0:
                del new_pair_cnt[pair]

        # 3. 计算新词汇 (与原代码一致)
        new_word = apply_merge(word_bytes, merge_pair)
        new_word_cnt[new_word] += cnt

        # 4. 查找新词汇的所有新对并更新 pair_cnt (与原代码一致)
        new_pairs = list(zip(new_word[:-1], new_word[1:]))
        for p in new_pairs:
            new_pair_cnt[p] += cnt
            
    return new_word_cnt, new_pair_cnt


def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """优化的 BPE 训练主函数 V2：移除 process_map 干扰。"""
    
    text = read_text(input_path)
    chunks = split_by_special(text, special_tokens, drop_special=True)

    # ----------------------------------------------------------------
    # 关键修改：直接使用 map 进行单进程处理，避免多进程卡死问题。
    # ----------------------------------------------------------------
    word_dicts = list(map(count_word, chunks))

    word_cnt = merge_dicts(word_dicts)
    pair_cnt = count_pair(word_cnt)

    vocab = get_basic_vocab(special_tokens)
    base_vocab_size = len(vocab)
    n_merges = vocab_size - base_vocab_size

    merges = []
    
    # 主训练循环
    for i in range(n_merges):
        # 1. 查找最大对
        if not pair_cnt:
            break
            
        max_pair = get_max_pair(pair_cnt)
        
        # 2. 更新词汇表和合并列表
        vocab[base_vocab_size + i] = max_pair[0] + max_pair[1]
        merges.append(max_pair)
        
        # 3. 核心优化：使用精确更新函数
        word_cnt, pair_cnt = update_cnt_optimized(word_cnt, pair_cnt, max_pair)
        
    return vocab, merges
