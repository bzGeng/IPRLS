import os
from collections import namedtuple

DATASETS = ['apparel', 'baby', 'books', 'camera_photo', 'dvd', 'electronics',
            'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines', 'MR',
            'music', 'software', 'sports_outdoors', 'toys_games', 'video']
OUT_DIR = "data/generated"
DATA_DIR = "data"
Raw_Example = namedtuple('Raw_Example', 'label task sentence')


def load_raw_data():
    for task_id, dataset in enumerate(DATASETS):
        yield _load_raw_data(dataset, task_id)


def _load_raw_data(dataset_name, task_id):
    train_file = os.path.join(DATA_DIR, dataset_name + '.task.train')
    train_data = _load_raw_data_from_file(train_file, task_id)
    test_file = os.path.join(DATA_DIR, dataset_name + '.task.test')
    test_data = _load_raw_data_from_file(test_file, task_id)
    return train_data, test_data


def _load_raw_data_from_file(filename, task_id):
    data = []
    with open(filename, encoding='utf-8', errors='ignore') as f:
        for line in f:
            segments = line.strip().split('\t')
            if len(segments) == 2:
                label = int(segments[0])
                tokens = segments[1].split(' ')
                example = Raw_Example(label, task_id, tokens)
                data.append(example)
    return data
