import re
from collections import Counter

# 读取OWT验证集（前100MB避免太慢）
with open('data/owt_valid.txt', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read(100_000_000)  # 前100MB

print(f"文件大小（采样）: {len(text):,} 字符\n")

# 1. 检查常见special tokens
special_tokens_to_check = [
    '<|endoftext|>',
    '<|startoftext|>',
    '<s>',
    '</s>',
    '<unk>',
    '<UNK>',
    '<pad>',
    '<mask>',
    '[CLS]',
    '[SEP]',
    '[PAD]',
    '[MASK]',
    '<BOS>',
    '<EOS>',
]

print("=== 检查常见Special Tokens ===")
for token in special_tokens_to_check:
    count = text.count(token)
    if count > 0:
        print(f"✅ 找到: {token} ({count} 次)")
    else:
        print(f"❌ 未找到: {token}")

# 2. 查找所有 <xxx> 或 [xxx] 模式
print("\n=== 所有 <xxx> 和 [xxx] 模式 ===")
angle_brackets = re.findall(r'<[^>]{1,30}>', text)
square_brackets = re.findall(r'\[[^\]]{1,30}\]', text)

if angle_brackets:
    counter = Counter(angle_brackets)
    print("尖括号模式（前20个）：")
    for token, count in counter.most_common(20):
        print(f"  {token}: {count}")
else:
    print("未找到尖括号模式")

if square_brackets:
    counter = Counter(square_brackets)
    print("\n方括号模式（前20个）：")
    for token, count in counter.most_common(20):
        print(f"  {token}: {count}")
else:
    print("未找到方括号模式")

# 3. 检查文档分隔符
print("\n=== 文档分隔模式 ===")
separators = [
    '\n\n',
    '\n\n\n',
    '---',
    '===',
    '***',
]
for sep in separators:
    count = text.count(sep)
    print(f"{repr(sep)}: {count:,} 次")

# 4. 统计基本信息
lines = text.split('\n')
print(f"\n=== 基本统计 ===")
print(f"总行数: {len(lines):,}")
print(f"空行数: {sum(1 for l in lines if not l.strip()):,}")
print(f"平均行长: {len(text)/len(lines):.1f} 字符")