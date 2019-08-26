
from pathlib import Path
import json

import tensorflow as tf

from main import model_fn

PARAMS = './results/params.json'
MODEL_DIR = './results/model'


def serving_input_receiver_fn():
    words = tf.placeholder(dtype=tf.string, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='nwords')
    receiver_tensors = {'words': words, 'nwords': nwords}
    features = {'words': words, 'nwords': nwords}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


if __name__ == '__main__':
    with Path(PARAMS).open(encoding='utf-8') as f:
        params = json.load(f)

    estimator = tf.estimator.Estimator(model_fn, MODEL_DIR, params=params)
    estimator.export_saved_model('saved_model', serving_input_receiver_fn)
