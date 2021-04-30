import random
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default='1',  help='')
    args = parser.parse_args()
    args.tasks = ['magazines', 'apparel', 'health_personal_care', 'camera_photo', 'toys_games', 'software', 'baby',
                  'kitchen_housewares', 'sports_outdoors',  'electronics', 'books', 'video', 'imdb', 'dvd', 'music', 'MR']
    random.seed(args.seed)
    random.shuffle(args.tasks)
    with open('shuffle_task_order.txt', 'w') as f:
        f.write('None' + '\n')
        for task in args.tasks:
            f.write(task + '\n')

