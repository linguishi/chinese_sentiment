import sys
import json
import logging
import functools
import numpy as np
import tensorflow as tf
from pathlib import Path

DATA_DIR = '../../data/hotel_comment'

# Logging
Path('results').mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


# Input function
def parse_fn(line_words, line_tag):
    # Encode in Bytes for TF
    words = [w.encode() for w in line_words.strip().split()]
    tag = line_tag.strip().encode()
    return (words, len(words)), tag


def generator_fn(words, tags):
    with Path(words).open('r', encoding='utf-8') as f_words, Path(tags).open('r', encoding='utf-8') as f_tags:
        for line_words, line_tag in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tag)


def input_fn(words_path, tags_path, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = (([None], ()), ())  # shape of every sample
    types = ((tf.string, tf.int32), tf.string)
    defaults = (('<pad>', 0), '')

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words_path, tags_path),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset


def model_fn(features, labels, mode, params):
    if isinstance(features, dict):
        features = features['words'], features['nwords']

    # Read vocabs and inputs
    dropout = params['dropout']
    words, nwords = features
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets'])
    with Path(params['tags']).open(encoding='utf-8') as f:
        indices = [idx for idx, tag in enumerate(f)]
        num_tags = len(indices)

    # Word Embeddings
    word_ids = vocab_words.lookup(words)
    w2v = np.load(params['w2v'])['embeddings']
    w2v_var = np.vstack([w2v, [[0.] * params['dim']]])
    w2v_var = tf.Variable(w2v_var, dtype=tf.float32, trainable=False)
    embeddings = tf.nn.embedding_lookup(w2v_var, word_ids)
    embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

    # LSTM
    t = tf.transpose(embeddings, perm=[1, 0, 2])
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    _, (cf, hf) = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    _, (cb, hb) = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    output = tf.concat([hf, hb], axis=-1)
    output = tf.layers.dropout(output, rate=dropout, training=training)

    # FC
    logits = tf.layers.dense(output, num_tags)
    pred_ids = tf.argmax(input=logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        reversed_tags = tf.contrib.lookup.index_to_string_table_from_file(params['tags'])
        pred_labels = reversed_tags.lookup(tf.argmax(input=logits, axis=1))
        predictions = {
            'classes_id': pred_ids,
            'labels': pred_labels
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    else:
        # LOSS
        tags_table = tf.contrib.lookup.index_table_from_file(params['tags'])
        tags = tags_table.lookup(labels)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=tags, logits=logits)

        # Metrics
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids),
            'precision': tf.metrics.precision(tags, pred_ids),
            'recall': tf.metrics.recall(tags, pred_ids)
        }

        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer().minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)


if __name__ == '__main__':
    params = {
        'dim': 300,
        'lstm_size': 32,
        'dropout': 0.5,
        'num_oov_buckets': 1,
        'epochs': 25,
        'batch_size': 20,
        'buffer': 3500,
        'words': str(Path(DATA_DIR, 'vocab.words.txt')),
        'tags': str(Path(DATA_DIR, 'vocab.labels.txt')),
        'w2v': str(Path(DATA_DIR, 'w2v.npz'))
    }

    with Path('results/params.json').open('w', encoding='utf-8') as f:
        json.dump(params, f, indent=4, sort_keys=True)


    def fwords(name):
        return str(Path(DATA_DIR, '{}.words.txt'.format(name)))


    def ftags(name):
        return str(Path(DATA_DIR, '{}.labels.txt'.format(name)))


    train_inpf = functools.partial(input_fn, fwords('train'), ftags('train'),
                                   params, shuffle_and_repeat=True)
    eval_inpf = functools.partial(input_fn, fwords('eval'), ftags('eval'))
    cfg = tf.estimator.RunConfig(save_checkpoints_secs=60)
    estimator = tf.estimator.Estimator(model_fn, 'results/model', cfg, params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=60)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


    # Write predictions to file
    def write_predictions(name):
        Path('results/score').mkdir(parents=True, exist_ok=True)
        with Path('results/score/{}.preds.txt'.format(name)).open('wb', encoding='utf-8') as f:
            test_inpf = functools.partial(input_fn, fwords(name), ftags(name))
            golds_gen = generator_fn(fwords(name), ftags(name))
            preds_gen = estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), tag) = golds
                f.write(b' '.join([tag, preds['labels'], ''.join(words)]) + b'\n')


    for name in ['train', 'eval']:
        write_predictions(name)
