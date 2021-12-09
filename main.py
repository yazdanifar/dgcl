#!/usr/bin/env python3
from argparse import ArgumentParser
import os
import yaml
import torch
from tensorboardX import SummaryWriter
from data import DataScheduler

# # Increase maximum number of open files from 1024 to 4096
# # as suggested in https://github.com/pytorch/pytorch/issues/973
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

parser = ArgumentParser()
parser.add_argument(
    '--episode', '-e', default='episodes/mnist_svhn-online.yaml'
)


def main():
    args = parser.parse_args()

    # Load config
    episode = yaml.load(open(args.episode), Loader=yaml.FullLoader)
    config = {'data_schedule': episode}

    # Build components
    data_scheduler = DataScheduler(config)


if __name__ == '__main__':
    main()
