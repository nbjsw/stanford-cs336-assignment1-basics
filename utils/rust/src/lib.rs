use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple, PyList};
use std::collections::HashMap;
use indicatif::{ProgressBar, ProgressStyle};

/// apply_merge - 内部辅助函数
fn apply_merge_internal(word_bytes: &[Vec<u8>], merge: &(Vec<u8>, Vec<u8>)) -> Vec<Vec<u8>> {
    let mut res = Vec::with_capacity(word_bytes.len());
    let merged = [merge.0.clone(), merge.1.clone()].concat();
    let mut i = 0;

    while i < word_bytes.len() {
        if i < word_bytes.len() - 1 && word_bytes[i] == merge.0 && word_bytes[i + 1] == merge.1 {
            res.push(merged.clone());
            i += 2;
        } else {
            res.push(word_bytes[i].clone());
            i += 1;
        }
    }
    res
}

/// Python 接口 apply_merge
#[pyfunction]
fn apply_merge(word_bytes: Vec<Vec<u8>>, merge: (Vec<u8>, Vec<u8>)) -> Vec<Vec<u8>> {
    apply_merge_internal(&word_bytes, &merge)
}

/// 找到最大 pair（保留 tie-break 逻辑）
fn get_max_pair(pair_cnt: &HashMap<(Vec<u8>, Vec<u8>), i32>) -> Option<(Vec<u8>, Vec<u8>)> {
    pair_cnt.iter()
        .max_by_key(|&(pair, &count)| (count, pair.clone()))
        .map(|(pair, _)| pair.clone())
}

/// 内部更新 word_cnt 和 pair_cnt
fn update_cnt_optimized_internal(
    word_cnt: HashMap<Vec<Vec<u8>>, i32>,
    mut pair_cnt: HashMap<(Vec<u8>, Vec<u8>), i32>,
    merge_pair: &(Vec<u8>, Vec<u8>),
) -> (HashMap<Vec<Vec<u8>>, i32>, HashMap<(Vec<u8>, Vec<u8>), i32>) {
    let mut new_word_cnt = HashMap::new();
    let mut words_to_update = Vec::new();

    // 分离需要更新和不需要更新的词
    for (word, cnt) in word_cnt {
        let contains_merge = word.windows(2).any(|w| w[0] == merge_pair.0 && w[1] == merge_pair.1);
        if contains_merge {
            words_to_update.push((word, cnt));
        } else {
            new_word_cnt.insert(word, cnt);
        }
    }

    // 更新包含 merge_pair 的词
    for (word, cnt) in words_to_update {
        // 减掉旧 pair 的计数
        for w in word.windows(2) {
            let key = (w[0].clone(), w[1].clone());
            if let Some(v) = pair_cnt.get_mut(&key) {
                *v -= cnt;
                if *v <= 0 {
                    pair_cnt.remove(&key);
                }
            }
        }

        // 应用 merge 得到新词
        let new_word = apply_merge_internal(&word, merge_pair);
        *new_word_cnt.entry(new_word.clone()).or_insert(0) += cnt;

        // 增加新 pair 的计数
        for w in new_word.windows(2) {
            *pair_cnt.entry((w[0].clone(), w[1].clone())).or_insert(0) += cnt;
        }
    }

    (new_word_cnt, pair_cnt)
}

/// Python 接口 update_cnt_optimized
#[pyfunction]
fn update_cnt_optimized<'py>(
    py: Python<'py>,
    word_cnt: HashMap<Vec<Vec<u8>>, i32>,
    pair_cnt: HashMap<(Vec<u8>, Vec<u8>), i32>,
    merge_pair: (Vec<u8>, Vec<u8>),
) -> PyResult<(&'py PyDict, &'py PyDict)> {

    let (new_word_cnt, new_pair_cnt) =
        update_cnt_optimized_internal(word_cnt, pair_cnt, &merge_pair);

    let py_word_cnt = PyDict::new(py);
    for (word, cnt) in new_word_cnt {
        let py_key = PyTuple::new(py, word.iter().map(|b| PyBytes::new(py, b)));
        py_word_cnt.set_item(py_key, cnt)?;
    }

    let py_pair_cnt = PyDict::new(py);
    for ((a, b), cnt) in new_pair_cnt {
        py_pair_cnt.set_item((PyBytes::new(py, &a), PyBytes::new(py, &b)), cnt)?;
    }

    Ok((py_word_cnt, py_pair_cnt))
}

/// BPE merge loop，保留 progress bar 和 tie-break
#[pyfunction]
fn bpe_merge_loop<'py>(
    py: Python<'py>,
    word_cnt: HashMap<Vec<Vec<u8>>, i32>,
    pair_cnt: HashMap<(Vec<u8>, Vec<u8>), i32>,
    n_merges: usize,
) -> PyResult<&'py PyList> {

    let mut current_word_cnt = word_cnt;
    let mut current_pair_cnt = pair_cnt;
    let mut merges = Vec::new();

    // 创建进度条
    let pb = ProgressBar::new(n_merges as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}] {per_sec}")
            .unwrap()
            .progress_chars("█▓▒░ ")
    );
    pb.set_message("BPE Merging");

    for _ in 0..n_merges {
        if current_pair_cnt.is_empty() { break; }

        let max_pair = match get_max_pair(&current_pair_cnt) {
            Some(p) => p,
            None => break,
        };
        merges.push(max_pair.clone());

        let (new_word_cnt, new_pair_cnt) = update_cnt_optimized_internal(
            current_word_cnt,
            current_pair_cnt,
            &max_pair
        );

        current_word_cnt = new_word_cnt;
        current_pair_cnt = new_pair_cnt;

        pb.inc(1);
    }

    pb.finish_with_message("BPE Merging complete");

    let py_merges = PyList::empty(py);
    for (a, b) in merges {
        py_merges.append((PyBytes::new(py, &a), PyBytes::new(py, &b)))?;
    }
    Ok(py_merges)
}

#[pymodule]
fn bpe_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_merge, m)?)?;
    m.add_function(wrap_pyfunction!(update_cnt_optimized, m)?)?;
    m.add_function(wrap_pyfunction!(bpe_merge_loop, m)?)?;
    Ok(())
}
