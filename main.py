from gensim.models import KeyedVectors

cn_word_vecs = KeyedVectors.load_word2vec_format('sgns.zhihu.bigram', binary=False)