import warnings
from abc import ABC, abstractmethod
from collections import Iterator, OrderedDict
from functools import reduce
import os
import math
import torch

from torch.utils.data import (
    Dataset,
    ConcatDataset,
    Subset,
    DataLoader,
    RandomSampler,
)

from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple

from torch.utils.data.dataloader import default_collate
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter


# =====================
# Base Classes and ABCs
# =====================


class DataScheduler(Iterator):
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.class_num = config['diva']['y_dim']
        self.domain_num = config['diva']['d_dim']
        self.schedule = config['data_schedule']
        self.datasets = OrderedDict()
        self.eval_datasets = OrderedDict()
        self.total_step = 0
        self.stage = -1

        self.domain_nums = []
        # Prepare training datasets
        for i, stage in enumerate(self.schedule['train']):
            stage_total = 0
            for j, subset in enumerate(stage['subsets']):
                dataset_name, subset_name = subset

                com_dataset_name = dataset_name
                if 'rotate' in stage:
                    com_dataset_name += str(stage['rotate'])

                if stage['domain'] not in self.domain_nums:
                    self.domain_nums.append(stage['domain'])
                if com_dataset_name in self.datasets:
                    stage_total += len(
                        self.datasets[com_dataset_name].subsets[subset_name])
                    continue
                self.datasets[com_dataset_name] = DATASET[dataset_name](config, stage)
                stage_total += len(
                    self.datasets[com_dataset_name].subsets[subset_name])

            if 'steps' in stage:
                self.total_step += stage['steps']
            elif 'epochs' in stage:
                self.total_step += int(
                    stage['epochs'] * (stage_total // config['batch_size']))
                if stage_total % config['batch_size'] > 0:
                    self.total_step += 1
            elif 'steps' in stage:
                self.total_step += sum(stage['steps'])
            else:
                self.total_step += stage_total // config['batch_size']

        # Prepare testing datasets
        if self.schedule['test']['tasks'] is None:
            self.schedule['test']['tasks'] = []

        # combine evaluation tasks
        if self.schedule['test']['include-training-task']:
            self.schedule['test']['tasks'] = self.schedule['train'] + self.schedule['test']['tasks']

        for i, stage in enumerate(self.schedule['test']['tasks']):
            for j, subset in enumerate(stage['subsets']):
                dataset_name, subset_name = subset

                com_dataset_name = dataset_name
                if 'rotate' in stage:
                    com_dataset_name += str(stage['rotate'])

                if stage['domain'] not in self.domain_nums:
                    self.domain_nums.append(stage['domain'])
                if com_dataset_name in self.eval_datasets:
                    continue
                self.eval_datasets[com_dataset_name] = DATASET[dataset_name](
                    config, stage, train=False
                )

        self.iterator = None
        self.task = None

        # check the domain dimension
        domain_dim_problem_message = "datasets has {} domains but model prepare for {} domains".format(
            max(self.domain_nums) + 1, self.domain_num)
        assert max(self.domain_nums) + 1 <= self.domain_num, domain_dim_problem_message

        if max(self.domain_nums) + 1 != self.domain_num:
            warnings.warn(domain_dim_problem_message)

    def __next__(self):
        try:
            if self.iterator is None:
                raise StopIteration
            data = next(self.iterator)
        except StopIteration:
            # Progress to next stage
            self.stage += 1
            print('\nProgressing to stage %d' % self.stage)
            if self.stage >= len(self.schedule['train']):
                raise StopIteration

            stage = self.schedule['train'][self.stage]
            self.task = stage['task']
            collate_fn = list(self.datasets.values())[0].collate_fn
            subsets = []
            for dataset_name, subset_name in stage['subsets']:

                com_dataset_name = dataset_name
                if 'rotate' in stage:
                    com_dataset_name += str(stage['rotate'])

                subsets.append(
                    self.datasets[com_dataset_name].subsets[subset_name])
            dataset = ConcatDataset(subsets)

            # Determine sampler
            if 'samples' in stage:
                sampler = RandomSampler(
                    dataset,
                    replacement=True,
                    num_samples=stage['samples']
                )
            elif 'steps' in stage:
                sampler = RandomSampler(
                    dataset,
                    replacement=True,
                    num_samples=stage['steps'] * self.config['batch_size']
                )
            elif 'epochs' in stage:
                sampler = RandomSampler(
                    dataset,
                    replacement=True,
                    num_samples=(int(stage['epochs'] * len(dataset))
                                 + len(dataset) % self.config['batch_size'])
                )
            else:
                sampler = RandomSampler(dataset)

            self.iterator = iter(DataLoader(
                dataset,
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_workers'],
                collate_fn=collate_fn,
                sampler=sampler,
                drop_last=True,
            ))

            data = next(self.iterator)

        # Get next data
        if self.task == "supervised":
            return data[0], data[1], data[2], self.stage
        elif self.task == "unsupervised":
            return data[0], None, data[2], self.stage
        else:
            print("task type is wrong it should be supervised or unsupervised but it is " + str(self.task))
            raise StopIteration

    def __len__(self):
        return self.total_step

    def eval_task(self, model, classifier_fn, writer, step, eval_title, task_id, description, data_loader, batch_size):
        model.eval()
        """
        compute the accuracy over the supervised training set or the testing set
        """
        predictions_d, actuals_d, predictions_y, actuals_y = [], [], [], []

        with torch.no_grad():
            # use the right data loader
            y_eye = torch.eye(self.class_num)
            d_eye = torch.eye(self.domain_num)
            for (xs, ys, ds) in data_loader:
                ys = y_eye[ys]

                # Convert to onehot

                ds = d_eye[ds]

                # To device
                xs, ys, ds = xs.to(self.device), ys.to(self.device), ds.to(self.device)

                # use classification function to compute all predictions for each batch
                pred_d, pred_y = classifier_fn(xs)
                predictions_d.append(pred_d)
                actuals_d.append(ds)
                predictions_y.append(pred_y)
                actuals_y.append(ys)

            # compute the number of accurate predictions
            accurate_preds_d = 0
            for pred, act in zip(predictions_d, actuals_d):
                for i in range(pred.size(0)):
                    v = torch.sum(pred[i] == act[i])
                    accurate_preds_d += (v.item() == self.domain_num)

            # calculate the accuracy between 0 and 1
            accuracy_d = (accurate_preds_d * 1.0) / (len(predictions_d) * batch_size)

            # compute the number of accurate predictions
            accurate_preds_y = 0
            for pred, act in zip(predictions_y, actuals_y):
                for i in range(pred.size(0)):
                    v = torch.sum(pred[i] == act[i])
                    accurate_preds_y += (v.item() == self.class_num)

            # calculate the accuracy between 0 and 1
            accuracy_y = (accurate_preds_y * 1.0) / (len(predictions_y) * batch_size)

            writer.add_scalar(
                'accuracy_y/%s/task_%s_%s' % (eval_title, str(task_id), description),
                accuracy_y, step
            )
            writer.add_scalar(
                'accuracy_d/%s/task_%s_%s' % (eval_title, str(task_id), description),
                accuracy_d, step
            )
            return accuracy_d, accuracy_y

    def get_dataloader(self, task_subsets, stage):
        collate_fn = list(self.datasets.values())[0].collate_fn
        subsets = []
        description = ""
        previous_dataset_name = None

        for dataset_name, subset_name in task_subsets:
            com_dataset_name = dataset_name
            if 'rotate' in stage:
                com_dataset_name += str(stage['rotate'])

            if previous_dataset_name is None or previous_dataset_name != com_dataset_name:
                description += com_dataset_name
            previous_dataset_name = com_dataset_name
            description += str(subset_name)

            subsets.append(
                self.eval_datasets[com_dataset_name].subsets[subset_name])
        dataset = ConcatDataset(subsets)

        # Determine sampler
        sampler = RandomSampler(dataset)

        eval_data_loader = DataLoader(
            dataset,
            batch_size=self.config['eval_batch_size'],
            num_workers=self.config['eval_num_workers'],
            collate_fn=collate_fn,
            sampler=sampler,
            drop_last=True,
        )
        return eval_data_loader, description

    def eval(self, model, classifier_fn, writer, step, eval_title):

        writer.add_scalar(
            'stage/%s' % (eval_title),
            self.stage, step
        )

        for i, stage in enumerate(self.schedule['test']['tasks']):
            eval_data_loader, description = self.get_dataloader(stage['subsets'], stage)
            accuracy_d, accuracy_y = self.eval_task(model, classifier_fn, writer, step, eval_title, i, description,
                                                    eval_data_loader,
                                                    self.config['batch_size'])


# ================
# Generic Datasets
# ================

class ClassificationDataset(Dataset, ABC):
    num_classes = NotImplemented
    targets = NotImplemented

    def __init__(self, config, stage, train=True):
        self.config = config
        self.subsets = {}
        self.train = train
        self.domain_id = stage['domain']

    def collate_fn(self, batch):
        return default_collate(batch)

    def offset_label(self):
        if 'label_offset' not in self.config:
            return

        if isinstance(self.targets, torch.Tensor):
            self.targets += self.config['label_offset'][self.name]
        else:
            for i in range(len(self.targets)):
                self.targets[i] += self.config['label_offset'][self.name]


# =================
# Concrete Datasets
# =================

class MNIST(torchvision.datasets.MNIST, ClassificationDataset):
    name = 'mnist'
    num_classes = 10

    def __init__(self, config, stage, train=True):

        # Compose transformation
        transform_list = [
            transforms.Resize((config['x_h'], config['x_w'])),
            transforms.ToTensor(),
        ]
        if config['recon_loss'] == 'bernoulli':
            transform_list.append(
                lambda x: (torch.rand_like(x) < x).to(torch.float)
            )
        if config['x_c'] > 1:
            transform_list.append(
                lambda x: x.expand(config['x_c'], -1, -1)
            )
        transform_list.append(
            transforms.RandomRotation(degrees=(stage['rotate'], stage['rotate']))
        )
        transform = transforms.Compose(transform_list)

        # Initialize super classes
        torchvision.datasets.MNIST.__init__(
            self, root=os.path.join(config['data_root'], 'mnist'),
            train=train, transform=transform, download=True)
        ClassificationDataset.__init__(self, config, stage, train)

        # Create subset for each class
        for y in range(self.num_classes):
            self.subsets[y] = Subset(
                self,
                list((self.targets == y).nonzero().squeeze(1).numpy())
            )
        self.offset_label()

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target = super(MNIST, self).__getitem__(index)
        return img, target, self.domain_id


class SVHN(torchvision.datasets.SVHN, ClassificationDataset):
    name = 'svhn'
    num_classes = 10

    def __init__(self, config, stage, train=True):
        # Compose transformation
        transform_list = [
            transforms.Resize((config['x_h'], config['x_w'])),
            transforms.Grayscale(num_output_channels=config['x_c']),
            transforms.ToTensor(),
        ]
        transform = transforms.Compose(transform_list)

        # Initialize super classes
        split = 'train' if train else 'test'
        torchvision.datasets.SVHN.__init__(
            self, root=os.path.join(config['data_root'], 'svhn'),
            split=split, transform=transform, download=True)
        ClassificationDataset.__init__(self, config, stage, train)

        # Create subset for each class
        self.targets = torch.Tensor(self.labels)
        for y in range(self.num_classes):
            self.subsets[y] = Subset(
                self,
                list((self.targets == y
                      ).nonzero().squeeze(1).numpy())
            )

        self.offset_label()

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target = super(SVHN, self).__getitem__(index)
        return img, target, self.domain_id


class CIFAR10(torchvision.datasets.CIFAR10, ClassificationDataset):
    name = 'cifar10'
    num_classes = 10

    def __init__(self, config, stage, train=True):
        if config.get('augment_cifar'):
            transform = transforms.Compose([
                transforms.Resize((config['x_h'], config['x_w'])),
                transforms.Grayscale(num_output_channels=config['x_c']),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((config['x_h'], config['x_w'])),
                torchvision.transforms.Grayscale(num_output_channels=config['x_c']),
                transforms.ToTensor(),
            ])
        torchvision.datasets.CIFAR10.__init__(
            self, root=os.path.join(config['data_root'], 'cifar10'),
            train=train, transform=transform, download=True)
        ClassificationDataset.__init__(self, config, stage, train)

        # Create subset for each class
        for y in range(self.num_classes):
            self.subsets[y] = Subset(
                self,
                list((torch.Tensor(self.targets) == y
                      ).nonzero().squeeze(1).numpy())
            )

        self.offset_label()

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target = super(CIFAR10, self).__getitem__(index)
        return img, target, self.domain_id


class USPS(torchvision.datasets.USPS, ClassificationDataset):
    name = 'usps'
    num_classes = 10

    def __init__(self, config, stage, train=True):
        if config.get('augment_usps'):
            transform = transforms.Compose([
                transforms.Resize((config['x_h'], config['x_w'])),
                transforms.Grayscale(num_output_channels=config['x_c']),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((config['x_h'], config['x_w'])),
                torchvision.transforms.Grayscale(num_output_channels=config['x_c']),
                transforms.ToTensor(),
            ])
        torchvision.datasets.USPS.__init__(
            self, root=os.path.join(config['data_root'], 'usps'),
            train=train, transform=transform, download=True)
        ClassificationDataset.__init__(self, config, stage, train)

        # Create subset for each class
        for y in range(self.num_classes):
            self.subsets[y] = Subset(
                self,
                list((torch.Tensor(self.targets) == y
                      ).nonzero().squeeze(1).numpy())
            )

        self.offset_label()

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target = super(USPS, self).__getitem__(index)
        return img, target, self.domain_id


class CIFAR100(torchvision.datasets.CIFAR100, ClassificationDataset):
    name = 'cifar100'
    num_classes = 100

    def __init__(self, config, stage, train=True):
        if config.get('augment_cifar'):
            transform = transforms.Compose([
                transforms.Resize((config['x_h'], config['x_w'])),
                transforms.Grayscale(num_output_channels=config['x_c']),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((config['x_h'], config['x_w'])),
                torchvision.transforms.Grayscale(num_output_channels=config['x_c']),
                transforms.ToTensor(),
            ])
        torchvision.datasets.CIFAR100.__init__(
            self, root=os.path.join(config['data_root'], 'cifar100'),
            train=train, transform=transform, download=True)
        ClassificationDataset.__init__(self, config, stage, train)

        # Create subset for each class
        for y in range(self.num_classes):
            self.subsets[y] = Subset(
                self,
                list((torch.Tensor(self.targets) == y
                      ).nonzero().squeeze(1).numpy())
            )

        self.offset_label()

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target = super(CIFAR100, self).__getitem__(index)
        return img, target, self.domain_id


DATASET = {
    MNIST.name: MNIST,
    SVHN.name: SVHN,
    USPS.name: USPS,
    CIFAR10.name: CIFAR10,
    CIFAR100.name: CIFAR100,
}
