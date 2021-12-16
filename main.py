#!/usr/bin/env python3
import datetime
import shutil
from argparse import ArgumentParser
import os
import yaml
import torch
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
    '--episode', '-e', default='episodes/mnist_svhn-online.yaml'
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
    current_time=datetime.datetime.now().strftime("_%H_%M_%S")
    config['log_dir'] = args.log_dir+current_time
    if os.path.exists(args.log_dir):
        print('WARNING: %s already exists' % args.log_dir)
        if not config['testing_mode']:
            input('Press enter to continue')
        shutil.rmtree(config['log_dir'])


    # Save config
    os.makedirs(config['log_dir'], mode=0o755, exist_ok=True)
    config_save_path = os.path.join(config['log_dir'], 'config.yaml')
    episode_save_path = os.path.join(config['log_dir'], 'episode.yaml')
    yaml.dump(config, open(config_save_path, 'w'))
    yaml.dump(episode, open(episode_save_path, 'w'))
    print('Config & episode saved to {}'.format(config['log_dir']))

    # Build components
    data_scheduler = DataScheduler(config)
    # test_dataset(data_scheduler)
    # return
    writer = SummaryWriter(config['log_dir'])
    model = MODEL[config['model_name']](config['diva'], config['batch_size'], writer)
    model.to(config['device'])
    train_model(config, model, data_scheduler, writer)


def test_dataset(scheduler):
    prev_t = -1
    for step, (x, y, d, t) in enumerate(scheduler):
        step += 1
        if prev_t != t:
            prev_t = t
            if y is None:
                print("X", x.shape, "Y is None", "d", d, "step", step, "task id", t)
            else:
                print("X", x.shape, "Y", y.shape, "d", d, "step", step, "task id", t)


if __name__ == '__main__':
    main()
