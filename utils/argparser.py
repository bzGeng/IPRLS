import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', type=str, default='',
                        help='Approach')
    parser.add_argument('--num_classes', type=int, default=-1,
                        help='Num outputs for dataset')

    # Optimization options.
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate for parameters, used for baselines')
    parser.add_argument('--lr_adapters', type=float, default=1e-4,
                        help='Learning rate for parameters, used for baselines')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=200,
                        help='input batch size for validation')
    parser.add_argument('--workers', type=int, default=24, help='')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')

    # Model configurations
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    parser.add_argument('--max_length', type=int, default=256, help='')
    parser.add_argument('--sample', default=True, help='')
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--gama', type=float, default=0.03)

    # Paths.
    parser.add_argument('--dataset', type=str, default='',
                        help='Name of dataset')
    parser.add_argument('--train_path', type=str, default='',
                        help='Location of train data')
    parser.add_argument('--val_path', type=str, default='',
                        help='Location of val data')
    parser.add_argument('--test_path', type=str, default='',
                        help='Location of test data')
    parser.add_argument('--save_prefix', type=str, default='checkpoints/',
                        help='Location to save model')
    parser.add_argument('--corpus_path', type=str, default='data',
                        help='Location of corpus')
    parser.add_argument('--acc_log_path', type=str, default='./acc.txt',
                        help='')

    # Other.
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--checkpoint_format', type=str,
                        default='{save_folder}/checkpoint-{epoch}.pth.tar',
                        help='checkpoint file format')
    parser.add_argument('--epochs', type=int, default=160,
                        help='number of epochs to train')
    parser.add_argument('--restore_epoch', type=int, default=0, help='')
    parser.add_argument('--save_folder', type=str,
                        help='folder name inside one_check folder')
    parser.add_argument('--load_folder', default='', help='')
    parser.add_argument('--one_shot_prune_perc', type=float, default=0.75,
                        help='% of neurons to prune per layer')
    parser.add_argument('--mode',
                        choices=['finetune', 'prune'],
                        help='Run mode')
    parser.add_argument('--log_path', type=str, help='')
    parser.add_argument('--total_num_tasks', type=int, help='')

    parser.add_argument('--Bert_path', type=str, default='Bert/', help='')

    # Else
    parser.add_argument('--build_data_seperate', default=False, help='')
    parser.add_argument('--build_data_file', default=False, help='')

    args = parser.parse_args()

    return args