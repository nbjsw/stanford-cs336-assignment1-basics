import pickle

with open('tokenizer/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

    print(f"Vocab size: {len(vocab)}")
    print(f"Max token ID in vocab: {max(vocab.values())}")
    print(f"Model vocab_size: 32768")

    # 检查是否有gap
    if max(vocab.values()) != len(vocab) - 1:
        print("⚠️ 警告：vocab中的ID不连续！")
