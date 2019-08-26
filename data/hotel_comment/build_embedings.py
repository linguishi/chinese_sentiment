from pathlib import Path
import numpy as np

if __name__ == '__main__':
    with Path('vocab.words.txt').open(encoding='utf-8') as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    with Path('vocab.words.txt').open(encoding='utf-8') as f:
        word_to_found = {line.strip(): False for line in f}

    size_vocab = len(word_to_idx)

    embeddings = np.zeros((size_vocab, 300))

    found = 0
    print('Reading W2V file (may take a while)')
    with Path('../../sgns.zhihu.bigram').open(encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line_idx % 100000 == 0:
                print('- At line {}'.format(line_idx))
            line = line.strip().split()
            if len(line) != 300 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if (word in word_to_idx) and (not word_to_found[word]):
                word_to_found[word] = True
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
    print('- done. Found {} vectors for {} words'.format(found, size_vocab))

    # 保存 np.array
    np.savez_compressed('w2v.npz', embeddings=embeddings)
