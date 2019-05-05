"""Reload and serve a saved model"""
import json
import jieba
from pathlib import Path
from tensorflow.contrib import predictor
from functools import partial

LINE = '''酒店设施不是新的，服务态度很不好'''


def predict(pred_fn, line):
    sentence = ' '.join(jieba.cut(line.strip(), cut_all=False, HMM=True))
    words = [w.encode() for w in sentence.strip().split()]
    nwords = len(words)
    predictions = pred_fn({'words': [words], 'nwords': [nwords]})
    return predictions


if __name__ == '__main__':
    export_dir = 'saved_model'
    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    predict_fn = partial(predict, predictor.from_saved_model(latest))
    print(LINE)
    print(predict_fn(LINE))
    line = input('\n\n输入一句中文： ')
    while line.strip().lower() != 'q':
        print('\n\n', predict_fn(line))
        line = input('\n\n输入一句中文： ')
