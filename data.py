import copy
import warnings
from abc import ABC, abstractmethod
from collections import Iterator, OrderedDict
from functools import reduce
import os
import math
import torch
import random

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

        self.datasets = [OrderedDict(), OrderedDict()]

        self.total_step = 0
        self.stage = -1

        self.domain_nums = []

        # Prepare training datasets
        for i, stage in enumerate(self.schedule['train']):
            stage_total = 0
            for j, subset in enumerate(stage['subsets']):
                dataset = self.get_subset_instance(subset)  # type:ProxyDataset
                if dataset.domain not in self.domain_nums:
                    self.domain_nums.append(dataset.domain)

                stage_total += len(dataset.inner_dataset)

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
            self.schedule['test']['tasks'] = self.schedule['train'] + DataScheduler.add_supervised_flag(
                self.schedule['test']['tasks'])

        for i, stage in enumerate(self.schedule['test']['tasks']):
            for j, subset in enumerate(stage['subsets']):
                dataset = self.get_subset_instance(subset, False)  # type:ProxyDataset
                if dataset.domain not in self.domain_nums:
                    self.domain_nums.append(dataset.domain)

        self.sup_iterator = None
        self.unsup_iterator = None
        self.task = None

        # check the domain dimension
        domain_dim_problem_message = "datasets has {} domains but model prepare for {} domains".format(
            max(self.domain_nums) + 1, self.domain_num)
        assert max(self.domain_nums) + 1 <= self.domain_num, domain_dim_problem_message

        if max(self.domain_nums) + 1 != self.domain_num:
            warnings.warn(domain_dim_problem_message)

    def get_subset_instance(self, subset, train=True):
        dataset_name, subset_name, domain, supervised, rotation = DataScheduler.get_subset_detail(subset)
        com_dataset_name = dataset_name + "|" + str(subset_name) + "|" + str(domain) + "|" + supervised + "|" + str(
            rotation)
        if com_dataset_name not in self.datasets[train]:
            new_dataset = ProxyDataset(self.config, subset, train)
            self.datasets[train][com_dataset_name] = new_dataset

        return self.datasets[train][com_dataset_name]

    @staticmethod
    def get_subset_detail(subset):
        try:
            dataset_name, subset_name, domain, supervised = subset
            return dataset_name, subset_name, domain, supervised, None
        except:
            dataset_name, subset_name, domain, supervised, rotation = subset
            return dataset_name, subset_name, domain, supervised, rotation

    @staticmethod
    def add_supervised_flag(tasks):
        tasks = copy.deepcopy(tasks)
        for i, stage in enumerate(tasks):
            for j, subset in enumerate(stage['subsets']):
                try:
                    dataset_name, subset_name, domain, rotation = subset
                    tasks[i]['subsets'][j] = (dataset_name, subset_name, domain, 'u', rotation)
                except:
                    dataset_name, subset_name, domain = subset
                    tasks[i]['subsets'][j] = (dataset_name, subset_name, domain, 'u')

        return tasks

    def get_data(self):
        if self.sup_iterator is None:
            if self.unsup_iterator is None:
                raise StopIteration
            else:
                data = next(self.unsup_iterator)
                unsup = True

        elif self.unsup_iterator is None:
            data = next(self.sup_iterator)
            unsup = False
        elif random.random() < self.unsup_portion:
            data = next(self.unsup_iterator)
            unsup = True
        else:
            data = next(self.sup_iterator)
            unsup = False
        return data, unsup

    def __next__(self):
        try:
            data, unsup = self.get_data()

        except StopIteration:
            # Progress to next stage
            self.stage += 1
            print('\nProgressing to stage %d' % self.stage)
            if self.stage >= len(self.schedule['train']):
                raise StopIteration

            stage = self.schedule['train'][self.stage]
            collate_fn = list(self.datasets[True].values())[0].collate_fn  # train = True
            sup_subsets = []
            unsup_subsets = []
            for j, subset in enumerate(stage['subsets']):
                dataset = self.get_subset_instance(subset)  # type:ProxyDataset

                if dataset.supervised == 's':
                    sup_subsets.append(dataset)
                elif dataset.supervised == 'u':
                    unsup_subsets.append(dataset)
                else:
                    print("supervised should be either s or u")

            sup_dataset = None
            unsup_dataset = None
            if len(sup_subsets) > 0:
                sup_dataset = ConcatDataset(sup_subsets)
                # print('sup dataset len', len(sup_dataset))
            if len(unsup_subsets) > 0:
                unsup_dataset = ConcatDataset(unsup_subsets)
                # print('unsup dataset len', len(unsup_dataset))

            # Determine sampler
            if 'samples' in stage:
                if sup_dataset is not None:
                    sup_sampler = RandomSampler(
                        sup_dataset,
                        replacement=True,
                        num_samples=stage['samples']
                    )
                if unsup_dataset is not None:
                    unsup_sampler = RandomSampler(
                        unsup_dataset,
                        replacement=True,
                        num_samples=stage['samples']
                    )
            elif 'steps' in stage:
                if sup_dataset is not None:
                    sup_sampler = RandomSampler(
                        sup_dataset,
                        replacement=True,
                        num_samples=stage['steps'] * self.config['batch_size']
                    )
                if unsup_dataset is not None:
                    unsup_sampler = RandomSampler(
                        unsup_dataset,
                        replacement=True,
                        num_samples=stage['steps'] * self.config['batch_size']
                    )
            elif 'epochs' in stage:
                if sup_dataset is not None:
                    sup_sampler = RandomSampler(
                        sup_dataset,
                        replacement=True,
                        num_samples=(int(stage['epochs'] * len(sup_dataset))
                                     + len(sup_dataset) % self.config['batch_size'])
                    )
                if unsup_dataset is not None:
                    unsup_sampler = RandomSampler(
                        unsup_dataset,
                        replacement=True,
                        num_samples=(int(stage['epochs'] * len(unsup_dataset))
                                     + len(unsup_dataset) % self.config['batch_size'])
                    )
            else:
                if sup_dataset is not None:
                    sup_sampler = RandomSampler(sup_dataset)
                if unsup_dataset is not None:
                    unsup_sampler = RandomSampler(unsup_dataset)
            if sup_dataset is not None:
                self.sup_iterator = iter(DataLoader(
                    sup_dataset,
                    batch_size=self.config['batch_size'],
                    num_workers=self.config['num_workers'],
                    collate_fn=collate_fn,
                    sampler=sup_sampler,
                    drop_last=True,
                ))
            if unsup_dataset is not None:
                self.unsup_iterator = iter(DataLoader(
                    unsup_dataset,
                    batch_size=self.config['batch_size'],
                    num_workers=self.config['num_workers'],
                    collate_fn=collate_fn,
                    sampler=unsup_sampler,
                    drop_last=True,
                ))

            if sup_dataset is None:
                self.unsup_portion = 1
            elif unsup_dataset is None:
                self.unsup_portion = 0
            else:
                self.unsup_portion = len(unsup_dataset) / (len(sup_dataset) + len(unsup_dataset))

            # print("in this stage unsup portion to all baches is:", self.unsup_portion)
            data, unsup = self.get_data()

        # Get next data
        if not unsup:
            return data[0], data[1], data[2], self.stage
        else:
            return data[0], None, data[2], self.stage

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

    def get_dataloader(self, stage):
        collate_fn = list(self.datasets[False].values())[0].collate_fn  # Train = False
        subsets = []
        description = ""
        previous_dataset_name = None

        for j, subset in enumerate(stage['subsets']):
            dataset = self.get_subset_instance(subset, False)  # type:ProxyDataset
            if previous_dataset_name is None or previous_dataset_name != dataset.complete_name:
                description += dataset.complete_name
            previous_dataset_name = dataset.complete_name
            description += str(dataset.subset_name)

            subsets.append(dataset)
        big_dataset = ConcatDataset(subsets)

        # Determine sampler
        sampler = RandomSampler(big_dataset)

        eval_data_loader = DataLoader(
            big_dataset,
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
            eval_data_loader, description = self.get_dataloader(stage)
            accuracy_d, accuracy_y = self.eval_task(model, classifier_fn, writer, step, eval_title, i, description,
                                                    eval_data_loader,
                                                    self.config['batch_size'])


# ================
# Generic Datasets
# ================
class ProxyDataset(Dataset):
    datasets = [{}, {}]

    def __init__(self, config, subset, train=True):
        dataset_name, subset_name, domain, supervised, rotation = DataScheduler.get_subset_detail(subset)
        self.domain = domain
        self.complete_name = dataset_name + str(domain) if rotation is None else str(rotation)
        self.supervised = supervised
        self.subset_name = subset_name

        transform_list = []
        if rotation is not None:
            transform_list.append(
                transforms.RandomRotation(degrees=(rotation, rotation))
            )

        self.transform = transforms.Compose(transform_list)
        if dataset_name not in ProxyDataset.datasets[train]:
            inner_dataset = DATASET[dataset_name](config, train=train)
            ProxyDataset.datasets[train][dataset_name] = inner_dataset
            self.inner_dataset = inner_dataset.subsets[subset_name]
        else:
            self.inner_dataset = ProxyDataset.datasets[train][dataset_name].subsets[subset_name]

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target = self.inner_dataset.__getitem__(index)
        if self.supervised:
            return self.transform(img), target, self.domain
        else:
            return self.transform(img), None, self.domain

    def collate_fn(self, batch):
        return default_collate(batch)


class ClassificationDataset(Dataset, ABC):
    num_classes = NotImplemented
    targets = NotImplemented

    def __init__(self, config, train=True):
        self.config = config
        self.subsets = {}
        self.train = train

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

    def __init__(self, config, train=True):

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
        transform = transforms.Compose(transform_list)

        # Initialize super classes
        torchvision.datasets.MNIST.__init__(
            self, root=os.path.join(config['data_root'], 'mnist'),
            train=train, transform=transform, download=True)
        ClassificationDataset.__init__(self, config, train)

        # Create subset for each class
        for y in range(self.num_classes):
            self.subsets[y] = Subset(
                self,
                list((self.targets == y).nonzero().squeeze(1).numpy())
            )
        self.offset_label()


class SVHN(torchvision.datasets.SVHN, ClassificationDataset):
    name = 'svhn'
    num_classes = 10

    def __init__(self, config, train=True):
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
        ClassificationDataset.__init__(self, config, train)

        # Create subset for each class
        self.targets = torch.Tensor(self.labels)
        for y in range(self.num_classes):
            self.subsets[y] = Subset(
                self,
                list((self.targets == y
                      ).nonzero().squeeze(1).numpy())
            )

        self.offset_label()


class CIFAR10(torchvision.datasets.CIFAR10, ClassificationDataset):
    name = 'cifar10'
    num_classes = 10

    def __init__(self, config, train=True):
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
        ClassificationDataset.__init__(self, config, train)

        # Create subset for each class
        for y in range(self.num_classes):
            self.subsets[y] = Subset(
                self,
                list((torch.Tensor(self.targets) == y
                      ).nonzero().squeeze(1).numpy())
            )

        self.offset_label()


class USPS(torchvision.datasets.USPS, ClassificationDataset):
    name = 'usps'
    num_classes = 10

    def __init__(self, config, train=True):
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
        ClassificationDataset.__init__(self, config, train)

        # Create subset for each class
        for y in range(self.num_classes):
            self.subsets[y] = Subset(
                self,
                list((torch.Tensor(self.targets) == y
                      ).nonzero().squeeze(1).numpy())
            )

        self.offset_label()


class CIFAR100(torchvision.datasets.CIFAR100, ClassificationDataset):
    name = 'cifar100'
    num_classes = 100

    def __init__(self, config, train=True):
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
        ClassificationDataset.__init__(self, config, train)

        # Create subset for each class
        for y in range(self.num_classes):
            self.subsets[y] = Subset(
                self,
                list((torch.Tensor(self.targets) == y
                      ).nonzero().squeeze(1).numpy())
            )

        self.offset_label()


DATASET = {
    MNIST.name: MNIST,
    SVHN.name: SVHN,
    USPS.name: USPS,
    CIFAR10.name: CIFAR10,
    CIFAR100.name: CIFAR100,
}
