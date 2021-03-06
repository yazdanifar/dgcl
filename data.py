import copy
import time
import warnings
from abc import ABC, abstractmethod
from collections import Iterator, OrderedDict
from functools import reduce
import os
import math

import numpy as np
import torch
import random
from scipy.io import loadmat

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
        self.this_task_epoch = 0
        self.total_epoch = 0
        self.current_task_epoch_size = 0

        self.sup_dataloader = None
        self.unsup_dataloader = None
        self.remained_epoch = 0

        self.config = config
        self.device = config['device']
        self.class_num = config['y_dim']
        self.domain_num = config['d_dim']
        self.schedule = config['data_schedule']

        self.datasets = [OrderedDict(), OrderedDict()]

        self.total_step = 0
        self.task_step = []
        self.stage = -1
        self.step = 0
        self.domain_nums = []

        # Prepare training datasets
        for i, stage in enumerate(self.schedule['train']):
            stage_total = 0
            for j, subset in enumerate(stage['subsets']):
                dataset = self.get_subset_instance(subset)  # type:ProxyDataset
                if dataset.domain not in self.domain_nums:
                    self.domain_nums.append(dataset.domain)

                stage_total += len(dataset)

            if 'epochs' in stage:
                stage_total = int(
                    stage['epochs'] * (stage_total // config['batch_size']))
                if stage_total % config['batch_size'] > 0:
                    stage_total += 1
            else:
                stage_total = stage_total // config['batch_size']

            self.total_step += stage_total
            print("task batches:", stage_total)
            self.task_step.append(stage_total)

        # Prepare testing datasets
        if self.schedule['test']['tasks'] is None:
            self.schedule['test']['tasks'] = []

        # add useless flags to eval dataset (supervised and portion)
        self.schedule['test']['tasks'] = DataScheduler.add_flags(self.schedule['test']['tasks'])

        # combine evaluation tasks
        if self.schedule['test']['include-training-task']:
            self.schedule['test']['tasks'] = self.schedule['train'] + self.schedule['test']['tasks']

        self.eval_data_loaders = []
        for i, stage in enumerate(self.schedule['test']['tasks']):
            for j, subset in enumerate(stage['subsets']):
                dataset = self.get_subset_instance(subset, False)  # type:ProxyDataset
                if dataset.domain not in self.domain_nums:
                    self.domain_nums.append(dataset.domain)
            eval_data_loader, description = self.get_dataloader(stage)
            self.eval_data_loaders.append((eval_data_loader, description, stage.get('start_from', 0)))

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
        dataset_name, subset_name, domain, supervised, portion, rotation = DataScheduler.get_subset_detail(subset)
        com_dataset_name = dataset_name + "|" + str(subset_name) + "|" + str(domain) + "|" + supervised + "|" + str(
            rotation)
        if com_dataset_name not in self.datasets[train]:
            new_dataset = ProxyDataset(self.config, subset, train)
            self.datasets[train][com_dataset_name] = new_dataset

        return self.datasets[train][com_dataset_name]

    @staticmethod
    def get_subset_detail(subset):
        try:
            dataset_name, subset_name, domain, supervised, portion = subset
            return dataset_name, subset_name, domain, supervised, portion, None
        except:
            dataset_name, subset_name, domain, supervised, portion, rotation = subset
            return dataset_name, subset_name, domain, supervised, portion, rotation

    @staticmethod
    def add_flags(tasks):
        tasks = copy.deepcopy(tasks)
        for i, stage in enumerate(tasks):
            for j, subset in enumerate(stage['subsets']):
                try:
                    dataset_name, subset_name, domain, rotation = subset
                    tasks[i]['subsets'][j] = (dataset_name, subset_name, domain, 'u', 1, rotation)
                except:
                    dataset_name, subset_name, domain = subset
                    tasks[i]['subsets'][j] = (dataset_name, subset_name, domain, 'u', 1)
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
        elif (self.unsup_portion > 0.5 and self.step % self.supervised_period != 1) or (
                self.unsup_portion < 0.5 and self.step % self.unsupervised_period == 1):
            data = next(self.unsup_iterator)
            unsup = True
        else:
            data = next(self.sup_iterator)
            unsup = False
        if self.step % self.current_task_epoch_size == self.current_task_epoch_size - 1:
            self.this_task_epoch += 1
            self.total_epoch += 1
        return data, unsup

    def stage_classes(self, stage_num, domain_id=None):
        ans = []
        if 0 <= stage_num < len(self.schedule['train']):
            stage = self.schedule['train'][stage_num]
            for j, subset in enumerate(stage['subsets']):
                dataset = self.get_subset_instance(subset, False)  # type:ProxyDataset
                if (dataset.domain, dataset.subset_name) not in ans and (
                        domain_id is None or dataset.domain == domain_id):
                    ans.append((dataset.domain, dataset.subset_name))
        return ans

    def __next__(self):
        try:
            self.step += 1
            data, unsup = self.get_data()
        except StopIteration:
            # Progress to next stage
            if self.remained_epoch > 0:
                print("next epoch, remained_epoch:", self.remained_epoch)
                self.remained_epoch -= 1
                if self.sup_dataloader is not None:
                    self.sup_iterator = iter(self.sup_dataloader)
                if self.unsup_dataloader is not None:
                    self.unsup_iterator = iter(self.unsup_dataloader)
            else:
                self.stage += 1
                self.step = 0
                print('\nProgressing to stage %d' % self.stage)
                if self.stage >= len(self.schedule['train']):
                    raise StopIteration

                stage = self.schedule['train'][self.stage]
                collate_fn = list(self.datasets[True].values())[0].collate_fn  # train = True
                sup_subsets = []
                unsup_subsets = []
                for j, subset in enumerate(stage['subsets']):
                    dataset = self.get_subset_instance(subset)  # type:ProxyDataset

                    if dataset.supervised:
                        sup_subsets.append(dataset)
                    else:
                        unsup_subsets.append(dataset)

                sup_dataset = None
                unsup_dataset = None
                if len(sup_subsets) > 0:
                    sup_dataset = ConcatDataset(sup_subsets)
                if len(unsup_subsets) > 0:
                    unsup_dataset = ConcatDataset(unsup_subsets)

                # Determine sampler
                if 'samples' in stage:
                    self.remained_epoch = 0
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
                elif 'epochs' in stage:
                    self.remained_epoch = stage['epochs']
                    sup_sampler = None
                    unsup_sampler = None
                else:
                    self.remained_epoch = 0
                    sup_sampler = None
                    unsup_sampler = None

                dataloader_kwargs = {"batch_size": self.config['batch_size'],
                                     "num_workers": self.config['num_workers'],
                                     "drop_last": True,
                                     "pin_memory": True,
                                     "collate_fn": collate_fn}

                self.this_task_epoch = 0
                self.current_task_epoch_size = 0
                if sup_dataset is not None:
                    if sup_sampler is None:
                        dataloader_kwargs["shuffle"] = True
                        self.sup_dataloader = DataLoader(
                            sup_dataset,
                            **dataloader_kwargs
                        )
                    else:
                        self.sup_dataloader = DataLoader(
                            sup_dataset,
                            **dataloader_kwargs
                        )
                    self.current_task_epoch_size += len(sup_dataset) // self.config['batch_size']
                    self.sup_iterator = iter(self.sup_dataloader)
                if unsup_dataset is not None:
                    if unsup_sampler is None:
                        dataloader_kwargs["shuffle"] = True
                        self.unsup_dataloader = DataLoader(
                            unsup_dataset,
                            **dataloader_kwargs
                        )
                    else:
                        self.unsup_dataloader = DataLoader(
                            unsup_dataset,
                            **dataloader_kwargs
                        )
                    self.current_task_epoch_size += len(unsup_dataset) // self.config['batch_size']
                    self.unsup_iterator = iter(self.unsup_dataloader)

                if sup_dataset is None:
                    self.unsup_portion = 1
                    self.supervised_period = 10000000000000000
                elif unsup_dataset is None:
                    self.unsup_portion = 0
                    self.unsupervised_period = 10000000000000000
                else:
                    self.unsup_portion = len(unsup_dataset) / (len(sup_dataset) + len(unsup_dataset))
                    self.supervised_period = 1
                    self.unsupervised_period = 1
                    if self.unsup_portion > 0.5:
                        self.supervised_period = int(1 / (1 - self.unsup_portion))
                        if self.supervised_period != 1 / (1 - self.unsup_portion):
                            print(self.unsup_portion,
                                  "we use periodic mode, unsupervised portion and supervised portion should castcade by an integer factor")
                            raise ValueError
                    else:
                        self.unsupervised_period = int(1 / self.unsup_portion)
                        if self.unsupervised_period != 1 / self.unsup_portion:
                            print(
                                "we use periodic mode, unsupervised portion and supervised portion should castcade by an integer factor")
                            raise ValueError
                print("in this stage unsup portion to all baches is:", round(self.unsup_portion, 3))

            data, unsup = self.get_data()

        # Get next data
        if not unsup:
            return data[0], data[1], data[2], self.stage
        else:
            return data[0], None, data[2], self.stage

    def __len__(self):
        return self.total_step

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
        # for evaluation no sampler is needed
        test_sampler = RandomSampler(big_dataset)
        eval_data_loader = DataLoader(
            big_dataset,
            batch_size=self.config['eval_batch_size'],
            num_workers=self.config['eval_num_workers'],
            collate_fn=collate_fn,
            sampler=test_sampler,
            drop_last=False,
            pin_memory=True
        )
        return eval_data_loader, description

    def eval_task(self, classifier_fn, writer, step, eval_title, task_id, description, data_loader):
        """
        compute the accuracy over the supervised training set or the testing set
        """
        accurate_preds_d = 0
        accurate_preds_y = 0
        with torch.no_grad():
            # use the right data loader
            y_eye = torch.eye(self.class_num, device=self.device)
            d_eye = torch.eye(self.domain_num, device=self.device)
            dataset_len = 0
            for inner_step, (xs, ys, ds) in enumerate(data_loader):
                dataset_len += ys.shape[0]
                # To device
                xs, ys, ds = xs.to(self.device).float(), ys.to(self.device).long(), ds.to(self.device).long()

                # Convert to onehot
                ds = d_eye[ds]
                ys = y_eye[ys]

                # use classification function to compute all predictions for each batch
                pred_d, pred_y = classifier_fn(xs)
                accurate_preds_d += torch.sum(pred_d * ds)
                accurate_preds_y += torch.sum(pred_y * ys)

            # calculate the accuracy between 0 and 1
            accuracy_d = (accurate_preds_d.item() * 1.0) / dataset_len

            # calculate the accuracy between 0 and 1
            accuracy_y = (accurate_preds_y.item() * 1.0) / dataset_len

            writer.add_scalar(
                'accuracy_d/task_%s_%s' % (str(task_id), description),
                accuracy_d, step
            )
            writer.add_scalar(
                'accuracy_y/task_%s_%s' % (str(task_id), description),
                accuracy_y, step
            )
            return accuracy_d, accuracy_y

    def eval(self, model, classifier_fn, writer, step, eval_title):
        starting_time = time.time()
        print("start evaluation", end=" ")
        writer.add_scalar(
            'stage/%s' % (eval_title),
            self.stage, step
        )
        model.eval()
        for i, eval_related in enumerate(self.eval_data_loaders):
            eval_data_loader, description, start_from = eval_related
            if self.stage >= start_from:
                print(f"\r go to stage {i} of evaluation", end=" ")
                accuracy_d, accuracy_y = self.eval_task(classifier_fn, writer, step, eval_title, i, description,
                                                        eval_data_loader)
        model.train()
        ending_time = time.time()
        print("\rTime elapsed:", ending_time - starting_time)


# ================
# Generic Datasets
# ================
class ProxyDataset(Dataset):
    datasets = [{}, {}]

    def __init__(self, config, subset, train=True):
        dataset_name, subset_name, domain, supervised, portion, rotation = DataScheduler.get_subset_detail(subset)
        self.domain = int(domain)
        self.complete_name = dataset_name + (str(domain) if rotation is None else str(rotation))
        self.supervised = (supervised == 's')
        self.subset_name = int(subset_name)
        self.portion = portion
        self.train = train
        self.black_and_white = (config['recon_loss'] == 'bernoulli')
        self.rotation = rotation
        if dataset_name not in ProxyDataset.datasets[train]:
            inner_dataset = DATASET[dataset_name](config, train=train)
            ProxyDataset.datasets[train][dataset_name] = inner_dataset

        self.inner_dataset = ProxyDataset.datasets[train][dataset_name].subsets[subset_name]

        if self.supervised:
            self.offset = 0
        else:
            self.offset = int(len(self.inner_dataset) * (1 - self.portion))

    def __len__(self):
        return int(len(self.inner_dataset) * self.portion)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target = self.inner_dataset.__getitem__(index + self.offset)
        if self.rotation is not None:
            img = transforms.functional.rotate(img, self.rotation)
        if self.black_and_white:
            img = (0.5 < img).to(torch.float)

        return img, target, self.domain

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
class MNISTSMALL(torchvision.datasets.MNIST, ClassificationDataset):
    name = 'mnist_small'
    num_classes = 10

    def __init__(self, config, train=True):

        # Compose transformation
        transform_list = [
            transforms.Resize((config['x_h'], config['x_w'])),
            transforms.ToTensor(),
        ]
        if config['x_c'] > 1:
            transform_list.append(
                lambda x: x.expand(config['x_c'], -1, -1)
            )
        transform = transforms.Compose(transform_list)

        # Initialize super classes
        torchvision.datasets.MNIST.__init__(
            self, root=os.path.join(config['data_root'], 'mnist'),
            train=True, transform=transform, download=True)
        ClassificationDataset.__init__(self, config, train)

        # Create subset for each class
        all_samples = list(np.load('rotated_mnist/supervised_inds_' + str(config['seed']) + '.npy'))
        for y in range(self.num_classes):
            list_samples_class_y = list((self.targets == y).nonzero().squeeze(1).numpy())
            list_samples = list(set(all_samples).intersection(list_samples_class_y))
            self.subsets[y] = Subset(
                self,
                list_samples
            )
        self.offset_label()


class MNIST(torchvision.datasets.MNIST, ClassificationDataset):
    name = 'mnist'
    num_classes = 10

    def __init__(self, config, train=True):

        # Compose transformation
        transform_list = [
            transforms.Resize((config['x_h'], config['x_w'])),
            transforms.ToTensor(),
        ]
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
            list_samples = list((self.targets == y).nonzero().squeeze(1).numpy())
            self.subsets[y] = Subset(
                self,
                list_samples
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


class CLOF(ClassificationDataset):
    name = 'CLOF'
    num_classes = 10

    def __init__(self, dataRoot, config, train=True):
        # Compose transformation
        transform_list = [
            transforms.ToTensor(),
        ]
        self.transform = transforms.Compose(transform_list)

        # Initialize super classes
        self.dataRoot = dataRoot

        allData = loadmat(self.dataRoot)  # ,variable_names='IMAGES',appendmat=True)#.get('IMAGES')

        self.labels = allData['labels'] - 1
        self.images = allData['feas']
        mean = np.mean(self.images, axis=0)
        self.images -= mean
        std = np.maximum(np.std(self.images, axis=0), 0.0001)
        self.images /= std

        l = 0 if 1 == 1 else int(len(self.labels) * 0.9)
        r = int(len(self.labels) * 0.9) if 1 == 1 else len(self.labels)

        self.labels = self.labels[l:r, 0]
        self.images = self.images[l:r]

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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.images[index], self.labels[index]
        img = np.expand_dims(img, axis=1)
        img = self.transform(img)
        return img, target

    def collate_fn(self, batch):
        return default_collate(batch)


class CLOFA(CLOF):
    name = 'CLOFA'
    num_classes = 10

    def __init__(self, config, train=True):
        super(CLOFA, self).__init__("data_files/caltech-ofiice-10/amazon_decaf.mat", config, train)


class CLOFC(CLOF):
    name = 'CLOFC'
    num_classes = 10

    def __init__(self, config, train=True):
        super(CLOFC, self).__init__("data_files/caltech-ofiice-10/caltech_decaf.mat", config, train)


class CLOFD(CLOF):
    name = 'CLOFD'
    num_classes = 10

    def __init__(self, config, train=True):
        super(CLOFD, self).__init__("data_files/caltech-ofiice-10/dslr_decaf.mat", config, train)


class CLOFW(CLOF):
    name = 'CLOFW'
    num_classes = 10

    def __init__(self, config, train=True):
        super(CLOFW, self).__init__("data_files/caltech-ofiice-10/webcam_decaf.mat", config, train)


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
    MNISTSMALL.name: MNISTSMALL,
    MNIST.name: MNIST,
    SVHN.name: SVHN,
    USPS.name: USPS,
    CIFAR10.name: CIFAR10,
    CIFAR100.name: CIFAR100,
    CLOFC.name: CLOFC,
    CLOFD.name: CLOFD,
    CLOFA.name: CLOFA,
    CLOFW.name: CLOFW,
}
