#!/usr/bin/env python3
import datetime
import shutil
from argparse import ArgumentParser
import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from data import DataScheduler
from models.model_diva import DIVA
from train import train_model

MODEL = {
    "diva": DIVA
    # "ndpm_model": NdpmModel
    # "our": OUR,
}

parser = ArgumentParser()
parser.add_argument(
    '--config', '-c', default='configs/super_diva_mnist.yaml'
)
parser.add_argument(
    '--episode', '-e', default='episodes/continual_diva_mnist_rotate.yaml'
    # 'episodes/diva_mnist_rotate_sup_and_unsup.yaml'
    # 'episodes/simple_mnist_for_test.yaml'
    # 'episodes/mnist_svhn-online.yaml'
)
parser.add_argument('--log-dir', '-l')
parser.add_argument('--override', default='')


def main():
    args = parser.parse_args()

    # Load config
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    episode = yaml.load(open(args.episode), Loader=yaml.FullLoader)
    config['data_schedule'] = episode

    # Override options
    for option in args.override.split('|'):
        if not option:
            continue
        address, value = option.split('=')
        keys = address.split('.')
        here = config
        for key in keys[:-1]:
            if key not in here:
                raise ValueError('{} is not defined in config file. '
                                 'Failed to override.'.format(address))
            here = here[key]
        if keys[-1] not in here:
            raise ValueError('{} is not defined in config file. '
                             'Failed to override.'.format(address))
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)

    # Set log directory
    if config['add_time_to_log']:
        current_time = datetime.datetime.now().strftime("_%H_%M_%S")
        config['log_dir'] = args.log_dir + current_time
    else:
        config['log_dir'] = args.log_dir

    if os.path.exists(config['log_dir']):
        print('WARNING: %s already exists' % config['log_dir'])
        if not config['testing_mode']:
            input('Press enter to continue')
        shutil.rmtree(config['log_dir'])

    # Save config
    os.makedirs(config['log_dir'], mode=0o755, exist_ok=True)
    config_save_path = os.path.join(config['log_dir'], 'config.yaml')
    episode_save_path = os.path.join(config['log_dir'], 'episode.yaml')
    model_save_path = os.path.join(config['log_dir'], 'model.pth')
    model_load_path = "logs/mnist_svhn_6/model.pth"

    yaml.dump(config, open(config_save_path, 'w'))
    yaml.dump(episode, open(episode_save_path, 'w'))
    print('Config & episode saved to {}'.format(config['log_dir']))

    # Build components
    data_scheduler = DataScheduler(config)
    # test_dataset(data_scheduler)
    # return
    writer = SummaryWriter(config['log_dir'])
    model = MODEL[config['model_name']](config['diva'], config['batch_size'], writer, config['device'])
    if model_load_path is not None:
        model.load_state_dict(torch.load(model_load_path))
        model.to(config['device'])
        for i in range(100):
            try:
                data_scheduler.learn_task(i)
            except:
                break
        train_model(config, model, data_scheduler, writer)

    else:
        model.to(config['device'])
        train_model(config, model, data_scheduler, writer)
        torch.save(model.state_dict(), model_save_path)

    test_generator(model, data_scheduler)


def test_generator(model: DIVA, dataset: DataScheduler):
    model.eval()
    sample_num = 10
    xx, yy, dd = model.get_replay_batch(dataset.learned_class, sample_num)
    # to cpu
    xx = xx.cpu()
    for i in range(sample_num):
        x, y, d = xx[i], yy[i], dd[i]

        plt.imshow(x.permute(1, 2, 0), cmap='gray')
        if y is None:
            print("X", x.shape, "Y is None", "d", d)
            plt.title(f" y = None   d={d}")
        else:
            print("X", x.shape, "Y", y, "d", d)
            plt.title(f" y = {y}   d={d}")

        plt.show()


def test_dataset(scheduler):
    prev_t = -1
    for step, (x, y, d, t) in enumerate(scheduler):
        step += 1
        if step < 20:  # prev_t != t:
            prev_t = t
            plt.imshow(x[0].permute(1, 2, 0), cmap='gray')
            if y is None:
                print("X", x.shape, "Y is None", "d", d.shape, "step", step, "task id", t)
                plt.title(f" y = None   d={d[0]}")
            else:
                print("X", x.shape, "Y", y.shape, "d", d.shape, "step", step, "task id", t)
                plt.title(f" y = {y[0]}   d={d[0]}")

            plt.show()


if __name__ == '__main__':
    main()
