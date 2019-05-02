# -*- coding: UTF-8 -*-

import os
import re
import numpy as np
import jieba
from gensim.models import KeyedVectors
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, LSTM
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# 导入词向量
cn_word_vecs = KeyedVectors.load_word2vec_format(
    'sgns.zhihu.bigram', binary=False)
ebd_dim = cn_word_vecs[u'我'].shape[0]

POS = os.path.join(os.getcwd(), 'fix_pos')
NEG = os.path.join(os.getcwd(), 'fix_neg')

# 存储所有评价，每例评价为一条string
train_text_raw = []
for file_name in os.listdir(POS):
    with open(os.path.join(POS, file_name), 'r') as f:
        train_text_raw.append(f.read().strip().decode('utf-8'))

for file_name in os.listdir(NEG):
    with open(os.path.join(NEG, file_name), 'r') as f:
        train_text_raw.append(f.read().strip().decode('utf-8'))

# train_tokens是一个长长的list，其中含有4000个小list，对应每一条评价
train_tokens = []
word_lists = []
pattern = ur"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
for text in train_text_raw:
    p_text = re.sub(pattern, "", text)
    cut = jieba.cut(p_text)
    cut_list = []
    word_list = []
    for word in cut:
        word_list.append(word)
        try:
            cut_list.append(cn_word_vecs.vocab[word].index)
        except KeyError:
            cut_list.append(0)
    train_tokens.append(cut_list)
    word_lists.append(word_list)

# 计算出最优长度
num_tokens = np.array([len(tokens) for tokens in train_tokens])
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)


def reverse_tokens(tokens):
    text = ''
    for i in tokens:
        if i != 0:
            text = text + cn_word_vecs.index2word[i]
        else:
            text = text + ' '
    return text


# 构造embeding
num_words = 50000
ebd_matrix = np.zeros((num_words, ebd_dim))
for i in range(num_words):
    ebd_matrix[i, :] = cn_word_vecs[cn_word_vecs.index2word[i]]
ebd_matrix = ebd_matrix.astype('float32')

# 处理不等长的序列
train_pad = pad_sequences(
    train_tokens, maxlen=max_tokens, padding='pre', truncating='pre')
train_pad[train_pad >= num_words] = 0
train_target = np.concatenate((np.ones(2000), np.zeros(2000)))

x_train, x_test, y_train, y_test = train_test_split(
    train_pad, train_target, test_size=0.1, random_state=12)

# 构造模型
model = Sequential()
model.add(
    Embedding(
        num_words,
        ebd_dim,
        weights=[ebd_matrix],
        input_length=max_tokens,
        trainable=False))
model.add(LSTM(units=16, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(lr=1e-3)
model.compile(
    loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

# 建立一个权重的存储点
path_checkpoint = 'sentiment_checkpoint.keras'
checkpoint = ModelCheckpoint(
    filepath=path_checkpoint,
    monitor='val_loss',
    verbose=1,
    save_weights_only=True,
    save_best_only=True)

# 定义early stoping如果3个epoch内validation loss没有改善则停止训练
earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# 自动降低learning rate
lr_reduction = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, min_lr=1e-5, patience=0, verbose=1)
# 定义callback函数
callbacks = [earlystopping, checkpoint, lr_reduction]

model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=128,
    callbacks=callbacks)

result = model.evaluate(x_test, y_test)
print 'Accuracy:%f' % result[1]


# 测试
def predict_sentiment(text):
    print text
    p_text = re.sub(pattern, "", text)
    cut = jieba.cut(p_text)
    cut_list = []
    for word in cut:
        try:
            cut_list.append(cn_word_vecs.vocab[word].index)
        except KeyError:
            cut_list.append(0)
    tokens_pad = pad_sequences([cut_list],
                               maxlen=max_tokens,
                               padding='pre',
                               truncating='pre')
    tokens_pad[tokens_pad >= num_words] = 0
    result = model.predict(x=tokens_pad)
    coef = result[0][0]
    if coef >= 0.5:
        print '是一例正面评价', 'output=%.2f' % coef
    else:
        print '是一例负面评价', 'output=%.2f' % coef


test_list = [
    '酒店设施不是新的，服务态度很不好', '酒店卫生条件非常不好', '床铺非常舒适', '房间很凉，不给开暖气', '房间很凉爽，空调冷气很足',
    '酒店环境不好，住宿体验很不好', '房间隔音不到位', '晚上回来发现没有打扫卫生', '因为过节所以要我临时加钱，比团购的价格贵'
]
for text in test_list:
    predict_sentiment(text)
