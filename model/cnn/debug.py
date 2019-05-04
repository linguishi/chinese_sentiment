from main import input_fn, model_fn, DATA_DIR
from pathlib import Path
import tensorflow as tf

tf.enable_eager_execution()

if __name__ == '__main__':
    params = {
        'dim': 300,
        'nwords': 10,
        'filter_sizes': [2, 3, 4, 5],
        'num_filters': 128,
        'dropout': 0.5,
        'num_oov_buckets': 1,
        'epochs': 25,
        'batch_size': 20,
        'buffer': 3500,
        'words': str(Path(DATA_DIR, 'vocab.words.txt')),
        'tags': str(Path(DATA_DIR, 'vocab.labels.txt')),
        'w2v': str(Path(DATA_DIR, 'w2v.npz'))
    }

    ds = input_fn(Path(DATA_DIR, 'train.words.txt'), Path(DATA_DIR, 'train.labels.txt'), params=params)
    iterator = ds.make_one_shot_iterator()
    batch_sample = iterator.get_next()
    model_fn(batch_sample[0], batch_sample[1], tf.estimator.ModeKeys.TRAIN, params)
