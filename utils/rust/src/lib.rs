use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use std::collections::{HashMap, HashSet, BinaryHeap};
use std::cmp::Ordering;
use indicatif::{ProgressBar, ProgressStyle};

// --- 数据结构 ---
#[derive(Debug, Clone, Eq, PartialEq)]
struct PairCount {
    count: i32,
    pair: (Vec<u8>, Vec<u8>),
}

impl Ord for PairCount {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count.cmp(&other.count)
            .then_with(|| self.pair.cmp(&other.pair))
    }
}

impl PartialOrd for PairCount {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// 使用 Struct 替代 HashMap Entry，支持原地修改
struct WordData {
    tokens: Vec<Vec<u8>>,
    count: i32,
}

// --- 核心逻辑 ---

#[pyfunction]
fn bpe_merge_loop<'py>(
    py: Python<'py>,
    word_cnt: HashMap<Vec<Vec<u8>>, i32>,
    _pair_cnt_unused: HashMap<(Vec<u8>, Vec<u8>), i32>, // 我们重新计算 pair_cnt 以确保索引一致
    n_merges: usize,
) -> PyResult<&'py PyList> {

    // 1. 初始化数据结构
    // 将 Python 传来的 Dict 扁平化为 Vec，方便通过索引访问
    let mut words: Vec<WordData> = word_cnt.into_iter().map(|(tokens, count)| {
        WordData { tokens, count }
    }).collect();

    // 倒排索引: Token -> [WordIndex, WordIndex, ...]
    // 告诉我们需要检查哪些单词
    let mut token_to_word_idx: HashMap<Vec<u8>, HashSet<usize>> = HashMap::new();
    let mut pair_counts: HashMap<(Vec<u8>, Vec<u8>), i32> = HashMap::new();
    let mut heap = BinaryHeap::new();

    // 2. 初始构建索引和 Pair 计数
    // 这一步虽然是全量扫描，但只做一次
    let pb_init = ProgressBar::new(words.len() as u64);
    pb_init.set_message("Building Index");
    
    for (idx, word) in words.iter().enumerate() {
        for i in 0..word.tokens.len() {
            let t = &word.tokens[i];
            token_to_word_idx.entry(t.clone()).or_default().insert(idx);

            if i < word.tokens.len() - 1 {
                let pair = (word.tokens[i].clone(), word.tokens[i+1].clone());
                *pair_counts.entry(pair).or_default() += word.count;
            }
        }
        pb_init.inc(1);
    }
    pb_init.finish_and_clear();

    // 初始建堆
    for (pair, &count) in pair_counts.iter() {
        heap.push(PairCount { count, pair: pair.clone() });
    }

    let mut merges = Vec::new();
    
    // 进度条
    let pb = ProgressBar::new(n_merges as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}] {per_sec}")
            .unwrap()
            .progress_chars("█▓▒░ ")
    );
    pb.set_message("BPE Merging");

    // 3. 主循环
    for _ in 0..n_merges {
        // --- Pop Max Pair ---
        let max_pair = loop {
            match heap.pop() {
                Some(pc) => {
                    match pair_counts.get(&pc.pair) {
                        Some(&real_count) if real_count == pc.count => break Some(pc.pair),
                        _ => continue, // Stale
                    }
                }
                None => break None,
            }
        };

        let best_pair = match max_pair {
            Some(p) => p,
            None => break,
        };

        merges.push(best_pair.clone());
        let merged_token = [best_pair.0.clone(), best_pair.1.clone()].concat();

        // --- 核心优化：只处理包含 best_pair.0 的单词 ---
        // 我们 clone 这个 HashSet 的一部分 keys (usize) 是非常廉价的
        // 必须 clone，因为我们要在循环中修改 token_to_word_idx
        let word_indices: Vec<usize> = match token_to_word_idx.get(&best_pair.0) {
            Some(indices) => indices.iter().cloned().collect(),
            None => Vec::new(),
        };

        let mut delta_map: HashMap<(Vec<u8>, Vec<u8>), i32> = HashMap::new();
        let (target_a, target_b) = &best_pair;

        for &idx in &word_indices {
            let word_data = &mut words[idx];
            let count = word_data.count;
            let tokens = &mut word_data.tokens;
            
            // 快速检查：如果单词里根本没有 target_b，那肯定无法合并 (A 后必须跟 B)
            // (虽然 A 在单词里，但可能是结尾，或者后面不是 B)
            // 这个检查能省去很多 vector 操作
            // 注意：这种检查在 Rust 里需要借用，稍微留意一下性能，这里直接做简单的循环逻辑即可
            
            let mut i = 0;
            let mut change_occurred = false;

            while i < tokens.len() {
                // 检查是否匹配 (A, B)
                if i < tokens.len() - 1 && &tokens[i] == target_a && &tokens[i+1] == target_b {
                    // --- Merge Logic ---
                    
                    // 1. 记录 Delta: 减少旧 Pair
                    *delta_map.entry((target_a.clone(), target_b.clone())).or_default() -= count;
                    
                    // 2. 处理左邻居 (Prev, A) -> (Prev, AB)
                    if i > 0 {
                        let prev = &tokens[i-1];
                        *delta_map.entry((prev.clone(), target_a.clone())).or_default() -= count;
                        *delta_map.entry((prev.clone(), merged_token.clone())).or_default() += count;
                    }
                    
                    // 3. 处理右邻居 (B, Next) -> (AB, Next)
                    if i + 2 < tokens.len() {
                        let next = &tokens[i+2];
                        *delta_map.entry((target_b.clone(), next.clone())).or_default() -= count;
                        *delta_map.entry((merged_token.clone(), next.clone())).or_default() += count;
                    }

                    // 4. 执行合并：修改 Vector
                    // 将 i 替换为 merged_token，删除 i+1
                    tokens[i] = merged_token.clone();
                    tokens.remove(i+1); // 这里的 remove 是 O(WordLength)，通常很短，可以接受
                    
                    change_occurred = true;
                    
                    // 这里的 i 不加，因为当前的 i 变成了 AB，可能和下一个 B 再次构成 AB B (如果逻辑允许，虽然BPE通常是从左到右 greedy)
                    // 标准 BPE 是 greedy left-to-right。
                    // 变成了 AB。我们需要检查 AB 是否和后面的 Token 构成新的一对？
                    // 不，当前轮次我们只合并 A+B。AB 是新 token，不会在这一轮再次被合并。
                    // 所以我们可以安全地跳过。
                    // 但是，如果原序列是 A B A B -> AB AB，我们需要继续处理后面的。
                    // 所以 i += 1 (跳过当前的 AB，去看下一个)
                    i += 1; 
                } else {
                    i += 1;
                }
            }

            if change_occurred {
                // 更新倒排索引：这个单词现在包含了 merged_token
                token_to_word_idx.entry(merged_token.clone()).or_default().insert(idx);
                // 注意：我们不需要从 A 或 B 的索引中删除 idx。
                // 1. 删除操作很慢 (HashSet remove)。
                // 2. 留着也无所谓，反正下次 pop A 时，代码会检查 word 里还有没有 A。如果没有，就什么都不做。
                // 这叫 "Lazy Index Cleanup"。
            }
        }

        // --- 应用 Delta 更新全局 Pair Counts ---
        for (pair, delta) in delta_map {
            let entry = pair_counts.entry(pair.clone()).or_insert(0);
            *entry += delta;
            
            if *entry > 0 {
                heap.push(PairCount { count: *entry, pair });
            } else {
                pair_counts.remove(&pair);
            }
        }
        
        pb.inc(1);
    }

    pb.finish_with_message("BPE Merging complete");

    // 4. 转换回 Python 列表
    let py_merges = PyList::empty(py);
    for (a, b) in merges {
        py_merges.append((PyBytes::new(py, &a), PyBytes::new(py, &b)))?;
    }
    Ok(py_merges)
}

// 定义一个 Rust 结构体来持有数据
#[pyclass]
struct CoreBPE {
    vocab: HashMap<Vec<u8>, i32>,
    merges: HashSet<(Vec<u8>, Vec<u8>)>,
}

#[pymethods]
impl CoreBPE {
    #[new]
    fn new(vocab: HashMap<Vec<u8>, i32>, merges: HashSet<(Vec<u8>, Vec<u8>)>) -> Self {
        CoreBPE { vocab, merges }
    }

    /// 核心逻辑：对应你 Python 中的 apply_merges
    /// 但是这里一次性处理一批单词，减少跨语言调用的开销
    fn encode_word_batch(&self, words: Vec<String>) -> Vec<i32> {
        let mut results = Vec::new();

        for word in words {
            // 1. 将单词转换为字节列表
            let mut current_bytes: Vec<Vec<u8>> = word.as_bytes()
                .iter()
                .map(|&b| vec![b])
                .collect();

            // 2. BPE Merge Loop
            loop {
                let mut best_pair_idx: Option<usize> = None;
                let mut min_token_id = i32::MAX;
                let mut best_merged_token: Vec<u8> = Vec::new();

                // 寻找最优合并对 (Greedy by lowest Token ID)
                // 对应 Python: if token_id < min_token_id
                if current_bytes.len() < 2 {
                    break;
                }

                for i in 0..current_bytes.len() - 1 {
                    let pair = (current_bytes[i].clone(), current_bytes[i + 1].clone());
                    
                    // 检查 pair 是否在 merges 集合中
                    if self.merges.contains(&pair) {
                        let combined = [pair.0, pair.1].concat();
                        
                        // 检查合并后的词是否在 vocab 中，并比较 ID
                        if let Some(&id) = self.vocab.get(&combined) {
                            if id < min_token_id {
                                min_token_id = id;
                                best_pair_idx = Some(i);
                                best_merged_token = combined;
                            }
                        }
                    }
                }

                // 执行合并
                match best_pair_idx {
                    Some(idx) => {
                        current_bytes[idx] = best_merged_token;
                        current_bytes.remove(idx + 1);
                    }
                    None => break, // 没有可合并的了
                }
            }

            // 3. 将最终的 bytes 转换为 ID
            for token_bytes in current_bytes {
                if let Some(&id) = self.vocab.get(&token_bytes) {
                    results.push(id);
                } else {
                    // 处理 UNK (虽然理论上字节级 BPE 不会有 UNK，但为了健壮性)
                    // 这里简单处理：忽略或报错，根据你的需求。
                    // 这里的逻辑假设所有 bytes 都在 vocab 里 (byte-level BPE 基础特性)
                    eprintln!("Warning: Byte sequence not found in vocab: {:?}", token_bytes);
                }
            }
        }

        results
    }
}


#[pymodule]
fn bpe_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    // 训练用的函数
    m.add_function(wrap_pyfunction!(bpe_merge_loop, m)?)?;
    // 推理用的类
    m.add_class::<CoreBPE>()?;
    Ok(())
}
