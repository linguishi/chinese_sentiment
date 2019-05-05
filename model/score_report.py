import argparse
from sklearn import metrics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('file', help='specify the path of the results file')
args = parser.parse_args()

if __name__ == '__main__':
    label_true = []
    label_pred = []
    target_names = []
    with Path(args.file).open() as f:
        for line in f:
            tag_name = line.strip().split()[0]
            if tag_name not in target_names:
                target_names.append(tag_name)
            label_true.append(tag_name)
            label_pred.append(line.strip().split()[1])
    print(metrics.classification_report(y_pred=label_pred, y_true=label_true, target_names=['POS', 'NEG']))
