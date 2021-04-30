import os
import torch
import random
from utils.load_data import load_raw_data
from transformers import RobertaTokenizer, BertTokenizer
PAD_WORD = "<pad>"
MAX_LEN = 500
DATASETS = ['apparel', 'baby', 'books', 'camera_photo', 'dvd', 'electronics',
            'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines', 'MR',
            'music', 'software', 'sports_outdoors', 'toys_games', 'video']
OUT_DIR = "./data/generated"
FILE_DIR = "./data/file"


def build_data_seperate():
    """
    load raw data
    """

    def _build_data(all_data):
        print('build data')
        for task_id, task_data in enumerate(all_data):
            train_data, test_data = task_data
            write_as_record(train_data, test_data, task_id)

    print('load raw data')
    all_data = []
    for task_data in load_raw_data():
        all_data.append(task_data)

    _build_data(all_data)


def convert_data(tokenizer, data_set, file, args):
    datas = []
    for item in data_set:
        data = {}
        data['label'] = torch.tensor(item['label'], dtype=torch.long)
        tokenizer_output = tokenizer(item['sentence'], padding='max_length', max_length=args.max_length,
                                     truncation=True)
        data['input_ids'] = torch.tensor(tokenizer_output['input_ids'])
        data['attention_mask'] = torch.tensor(tokenizer_output['attention_mask'])
        datas.append(data)
    torch.save(datas, file)


def build_data_file(args):
    """
    load raw data
    """
    if args.PTM == 'RoBerta':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    elif args.PTM == 'Bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_set = torch.load(os.path.join(OUT_DIR, args.dataset + '.train.record'))
    val_set = torch.load(os.path.join(OUT_DIR, args.dataset + '.val.record'))
    test_set = torch.load(os.path.join(OUT_DIR, args.dataset + '.test.record'))
    train_file = os.path.join(FILE_DIR, args.dataset + '.train.file')
    val_file = os.path.join(FILE_DIR, args.dataset + '.val.file')
    test_file = os.path.join(FILE_DIR, args.dataset + '.test.file')
    convert_data(tokenizer, train_set, train_file, args)
    convert_data(tokenizer, val_set, val_file, args)
    convert_data(tokenizer, test_set, test_file, args)


def write_as_record(train_data, test_data, task_id):
    dataset = DATASETS[task_id]
    train_record_file = os.path.join(OUT_DIR, dataset + '.train.record')
    val_record_file = os.path.join(OUT_DIR, dataset + '.val.record')
    test_record_file = os.path.join(OUT_DIR, dataset + '.test.record')

    write_to_file_train(train_data, train_record_file, val_record_file)
    write_to_file_test(test_data, test_record_file)


def write_to_file_train(raw_data, train_record_file, val_record_file ):
    """convert the raw data to TXT

    Args:
      raw_data: a list of Raw_Example
      filename: file to write in
    """
    random.shuffle(raw_data)
    train_examples = []
    for raw_example in raw_data[:-200]:
        raw_example = raw_example._asdict()
        raw_example['sentence'] = ' '.join(raw_example['sentence'])
        train_examples.append(raw_example)

    torch.save(train_examples, train_record_file)

    val_examples = []
    for raw_example in raw_data[-200:]:
        raw_example = raw_example._asdict()
        raw_example['sentence'] = ' '.join(raw_example['sentence'])
        val_examples.append(raw_example)
    torch.save(val_examples, val_record_file)


def write_to_file_test(raw_data, test_record_file):
    """convert the test data to TXT

    Args:
      raw_data: a list of Raw_Example
      test_record_file: file to write in
    """
    examples = []
    for raw_example in raw_data:
        raw_example = raw_example._asdict()
        raw_example['sentence'] = ' '.join(raw_example['sentence'])
        examples.append(raw_example)
    torch.save(examples, test_record_file)



