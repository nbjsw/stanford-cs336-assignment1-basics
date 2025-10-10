import regex as re
from collections import defaultdict
from tqdm.contrib.concurrent import process_map

# ----------------------------------------------------------------------
# 核心 BPE 训练函数
# ----------------------------------------------------------------------
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def read_text(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def split_by_special(text, special_tokens, drop_special=True):
    if not special_tokens:
        return [text]

    # Sort by descending length to prioritize longer tokens (e.g., "<|endoftext|><|endoftext|>" before "<|endoftext|>")
    special_tokens = sorted(special_tokens, key=len, reverse=True)

    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if not drop_special: pattern = f"({pattern})"

    pattern = re.compile(pattern)
    chunks = pattern.split(text)
    return [c for c in chunks if c]  # remove empty strings


def word2bytes(word):
    "Convert word string to tuple of bytes"
    a = list(word.encode('utf-8'))
    return tuple(bytes([i]) for i in a)


def count_word(text):
    "Split text into word bytes using GPT2 pattern and count word bytes frequency."
    word_cnt = defaultdict(int)
    for m in PAT.finditer(text):
        word = m.group(0)
        word_bytes = word2bytes(word)
        if len(word_bytes)>=2:
            word_cnt[word_bytes]+=1
    return word_cnt


def merge_dicts(dicts):
    merged = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            merged[k] += v
    return merged


def count_pair(word_cnt):
    pair_cnt = defaultdict(int)
    for word_bytes,cnt in word_cnt.items():
        for pair in zip(word_bytes[:-1],word_bytes[1:]):
            pair_cnt[pair]+=cnt
    return pair_cnt


def get_max_pair(pair_cnt):
    max_pair, _ = max(pair_cnt.items(), key=lambda x: (x[1], x[0]))  # lexicographic tie-breaker
    return max_pair


def get_basic_vocab(special_tokens):
    vocab={token:bytes([token]) for token in range(256)}

    for i,token in enumerate(special_tokens):
        token_id = 256+i
        vocab[token_id] = token.encode("utf-8")
    return vocab


def apply_merge(word_bytes,merge):
    merged = merge[0]+merge[1]
    i = 0
    new_word_bytes = []
    while i < len(word_bytes):
        # Check for match
        if i < len(word_bytes) - 1 and word_bytes[i] == merge[0] and word_bytes[i+1] == merge[1]:
            new_word_bytes.append(merged)
            i += 2
        else:
            new_word_bytes.append(word_bytes[i])
            i += 1
    return tuple(new_word_bytes)


def update_cnt(word_cnt,pair_cnt, merge_pair):

    new_word_cnt = defaultdict(int)
    new_pair_cnt = defaultdict(int, pair_cnt) # copy with defaultdict

    for word_bytes,cnt in word_cnt.items():

        #----------for word cnt ---------------

        old_pairs = list(zip(word_bytes[:-1], word_bytes[1:]))

        # Keep the original count if the merge not appear in the key
        if merge_pair not in old_pairs:
            new_word_cnt[word_bytes]+=cnt
            continue

        # Use updated key if merge appear
        new_word = apply_merge(word_bytes,merge_pair)
        new_word_cnt[new_word]+=cnt

        #--------for pair cnt ----------------

        # Decrease all old pair counts
        for pair in old_pairs:
            new_pair_cnt[pair]-=cnt
            if new_pair_cnt[pair] ==0:
                del new_pair_cnt[pair]

        # Count new pairs in the new word
        new_pairs = list(zip(new_word[:-1], new_word[1:]))
        for p in new_pairs:
            new_pair_cnt[p] += cnt

    return new_word_cnt,new_pair_cnt


def train_bpe(input_path,vocab_size,special_tokens):
    text = read_text(input_path)
    chunks = split_by_special(text,special_tokens)

    # Only parallelize if chunk count is big enough
    if len(chunks) < 4: word_dicts = list(map(count_word, chunks))
    else: word_dicts = process_map(count_word, chunks, chunksize=1)

    word_cnt = merge_dicts(word_dicts)
    pair_cnt = count_pair(word_cnt)

    vocab = get_basic_vocab(special_tokens)
    base_vocab_size = len(vocab)
    n_merges=vocab_size-base_vocab_size

    merges = []
    for i in range(n_merges):
        max_pair = get_max_pair(pair_cnt)
        vocab[base_vocab_size+i] = max_pair[0]+max_pair[1]
        merges.append(max_pair)
        word_cnt, pair_cnt = update_cnt(word_cnt,pair_cnt,max_pair)
    return vocab, merges


def train_bpe_no_pretokenization(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. 初始化词汇表
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    # 确定下一个可用的 ID，从 256 开始
    next_id = 256

    # 2. 处理特殊标记
    for token in special_tokens:
        if token.encode('utf-8') in vocab.values():
             # 确保特殊标记没有和现有词汇冲突 (虽然 unlikely for the initial 256 bytes)
             continue
        vocab[next_id] = token.encode('utf-8')
        next_id += 1

    # 计算需要执行的合并次数
    num_merges = vocab_size - next_id
    if num_merges <= 0:
        # 如果初始词汇量已经达到或超过目标，则直接返回
        return vocab, merges

    # 3. 数据预处理（获取训练数据块）
    with open(input_path, 'rb') as f:
        data_bytes = f.read()

    # 简化的预分词（仅按空格和标点分割，但 BBPE 最好使用 GPT-2 风格的预分词）
    # 生产环境中：使用复杂的 re_engine.findall(GPT2_REGEX, data_string)
    # 教学/测试环境中：简单的字节序列

    # 使用简单的字节序列作为训练数据（每个元素是初始字节的 ID，长度为文件字节长度）
    # 更好的方法是使用预分词，将数据分割成 ID 列表的列表 (List[List[int]])
    
    # 我们将训练数据表示为字典: {word_tuple: count}
    # 初始时，每个“word”是单个字节 ID 的元组
    training_data: defaultdict[tuple[int, ...], int] = collections.defaultdict(int)

    # 假设没有预分词，整个文件是一个巨大的字节序列
    # 为了效率，这里假设数据已经通过某种方式（例如空格/GPT-2正则）分割成“单词”
    # 简化：假设每个字节都是一个初始标记

    # 在没有复杂预分词的情况下，将整个文件内容视为一个长序列（性能瓶颈点）
    # For robust, competitive implementation, you should use the pre-tokenization step.
    # 假设 data_bytes 已经被预处理成多个块（例如，按空格分割的单词的字节表示）
    # Simplest: treat all adjacent bytes as potential pairs

    # Step 3b: 将原始字节序列转换为初始 token ID 列表
    # (如果使用预分词，则是一个列表的列表)
    token_id_sequence = list(data_bytes) # [b0, b1, b2, ...] -> [id0, id1, id2, ...]

    # 4. 训练循环
    for merge_step in range(num_merges):
        # 4a. 统计相邻对频率
        pair_counts: defaultdict[Tuple[int, int], int] = collections.defaultdict(int)
        
        # 遍历序列，统计相邻 token ID 对
        for i in range(len(token_id_sequence) - 1):
            pair = (token_id_sequence[i], token_id_sequence[i+1])
            pair_counts[pair] += 1
        
        if not pair_counts:
            # 没有更多的对可以合并，停止训练
            break
            
        # 4b. 找到最高频对
        # 找到频率最高的 pair，如果有平局，选择字典序最小的 pair
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
        
        # 4c. 创建新标记和记录合并
        id1, id2 = best_pair
        new_id = next_id
        
        # 查找合并前的字节表示
        bytes1 = vocab[id1]
        bytes2 = vocab[id2]
        new_token_bytes = bytes1 + bytes2
        
        # 记录合并规则
        merges.append((bytes1, bytes2))
        
        # 更新词汇表
        vocab[new_id] = new_token_bytes
        next_id += 1
        
        # 4d. 更新训练数据（执行替换）
        # 使用新 ID 替换 token_id_sequence 中所有出现的 (id1, id2)
        
        new_sequence = []
        i = 0
        while i < len(token_id_sequence):
            # 检查当前位置 i 和 i+1 是否是最佳合并对
            if i + 1 < len(token_id_sequence) and token_id_sequence[i] == id1 and token_id_sequence[i+1] == id2:
                new_sequence.append(new_id)
                i += 2 # 跳过第二个元素，因为它被合并了
            else:
                new_sequence.append(token_id_sequence[i])
                i += 1
        
        token_id_sequence = new_sequence
        
    # 5. 返回结果
    return vocab, merges

