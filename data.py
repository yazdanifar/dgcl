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

        self.domain_num_count = 0
        # Prepare training datasets
        for i, stage in enumerate(self.schedule['train']):
            stage_total = 0
            for j, subset in enumerate(stage['subsets']):
                dataset_name, subset_name = subset
                if dataset_name in self.datasets:
                    stage_total += len(
                        self.datasets[dataset_name].subsets[subset_name])
                    continue
                self.datasets[dataset_name] = DATASET[dataset_name](config, self.domain_num_count)
                if self.schedule['test']['include-training-task']:
                    self.eval_datasets[dataset_name] = DATASET[dataset_name](
                        config, self.domain_num_count, train=False
                    )
                self.domain_num_count += 1
                stage_total += len(
                    self.datasets[dataset_name].subsets[subset_name])

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
        if self.schedule['test']['tasks'] is not None:
            for i, stage in enumerate(self.schedule['test']['tasks']):
                for j, subset in enumerate(stage['subsets']):
                    dataset_name, subset_name = subset
                    if dataset_name in self.eval_datasets:
                        continue
                    self.eval_datasets[dataset_name] = DATASET[dataset_name](
                        config, self.domain_num_count, train=False
                    )
                    self.domain_num_count += 1

        self.iterator = None
        self.task = None

        # check the domain dimension

        domain_dim_problem_message = "datasets has {} domains but model prepare for {} domains".format(
            self.domain_num_count, self.domain_num)
        assert self.domain_num_count < self.domain_num, domain_dim_problem_message

        if self.domain_num_count != self.domain_num:
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
                subsets.append(
                    self.datasets[dataset_name].subsets[subset_name])
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

    def eval_task(self, model, classifier_fn, writer, step, eval_title, task_id, data_loader, batch_size):
        model.eval()  # TODO: what is this?
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
                    accurate_preds_d += (v.item() == 5)

            # calculate the accuracy between 0 and 1
            accuracy_d = (accurate_preds_d * 1.0) / (len(predictions_d) * batch_size)

            # compute the number of accurate predictions
            accurate_preds_y = 0
            for pred, act in zip(predictions_y, actuals_y):
                for i in range(pred.size(0)):
                    v = torch.sum(pred[i] == act[i])
                    accurate_preds_y += (v.item() == 10)

            # calculate the accuracy between 0 and 1
            accuracy_y = (accurate_preds_y * 1.0) / (len(predictions_y) * batch_size)

            return accuracy_d, accuracy_y

    def eval(self, model, classifier_fn, writer, step, eval_title):
        if self.schedule['test']['include-training-task']:
            for i, stage in enumerate(self.schedule['train']):
                # shouldn't use self.task and self.stage
                task = stage['task']
                collate_fn = list(self.datasets.values())[0].collate_fn
                subsets = []
                for dataset_name, subset_name in stage['subsets']:
                    subsets.append(
                        self.eval_datasets[dataset_name].subsets[subset_name])
                dataset = ConcatDataset(subsets)

                # Determine sampler
                sampler = RandomSampler(dataset)

                eval_data_loader = DataLoader(
                    dataset,
                    batch_size=self.config['batch_size'],
                    num_workers=self.config['num_workers'],
                    collate_fn=collate_fn,
                    sampler=sampler,
                    drop_last=True,
                )
                self.eval_task(model, classifier_fn, writer, step, eval_title, i, eval_data_loader,
                               self.config['batch_size'])


class BaseDataset(Dataset, ABC):
    name = 'base'

    def __init__(self, config, train=True):
        self.config = config
        self.subsets = {}
        self.train = train

    def eval(self, model, writer: SummaryWriter, step, eval_title,
             task_index=None):
        if self.config['eval_d']:
            self._eval_discriminative_model(model, writer, step, eval_title)
        if self.config['eval_g']:
            self._eval_generative_model(model, writer, step, eval_title)
        if 'eval_t' in self.config and self.config['eval_t']:
            self._eval_hard_assign(
                model, writer, step, eval_title,
                task_index=task_index
            )

    @abstractmethod
    def _eval_hard_assign(
            self,
            model,
            writer: SummaryWriter,
            step, eval_titlem, task_index=None):
        raise NotImplementedError

    @abstractmethod
    def _eval_discriminative_model(
            self,
            model,
            writer: SummaryWriter,
            step, eval_title):
        raise NotImplementedError

    @abstractmethod
    def _eval_generative_model(
            self,
            model,
            writer: SummaryWriter,
            step, eval_title):
        raise NotImplementedError

    def collate_fn(self, batch):
        return default_collate(batch)


class CustomSubset(Subset):
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.dataset[self.indices[idx]])


# ================
# Generic Datasets
# ================

class ClassificationDataset(BaseDataset, ABC):
    num_classes = NotImplemented
    targets = NotImplemented

    def __init__(self, config, train=True):
        super().__init__(config, train)

    def _eval_hard_assign(
            self,
            model,
            writer: SummaryWriter,
            step, eval_title, task_index=None,
    ):
        tasks = [
            tuple([c for _, c in t['subsets']])
            for t in self.config['data_schedule']
        ]
        if task_index is not None:
            tasks = [tasks[task_index]]
        k = 5

        # Overall counts
        total_overall = 0.
        correct_1_overall = 0.
        correct_k_overall = 0.
        correct_expert_overall = 0.
        correct_assign_overall = 0.

        # Loop over each task
        for task_index, task_subsets in enumerate(tasks, task_index or 0):
            # Task-wise counts
            total = 0.
            correct_1 = 0.
            correct_k = 0.
            correct_expert = 0.
            correct_assign = 0.

            # Loop over each subset
            for subset in task_subsets:
                data = DataLoader(
                    self.subsets[subset],
                    batch_size=self.config['eval_batch_size'],
                    num_workers=self.config['eval_num_workers'],
                    collate_fn=self.collate_fn,
                )
                for x, y in iter(data):
                    with torch.no_grad():
                        logits, assignments = model(
                            x, return_assignments=True)
                    total += x.size(0)
                    correct_assign += (assignments == task_index).float().sum()
                    if not self.config['disable_d']:
                        # NDPM accuracy
                        _, pred_topk = logits.topk(k, dim=1)
                        correct_topk = (
                                pred_topk.cpu()
                                == y.unsqueeze(1).expand_as(pred_topk)
                        ).float()
                        correct_1 += correct_topk[:, :1].view(-1).sum()
                        correct_k += correct_topk[:, :k].view(-1).sum()

                        # Hard-assigned expert accuracy
                        num_experts = len(model.ndpm.experts) - 1
                        if num_experts > task_index:
                            expert = model.ndpm.experts[task_index + 1]
                            with torch.no_grad():
                                logits = expert(x)
                            correct = (y == logits.argmax(dim=1).cpu()).float()
                            correct_expert += correct.sum()

            # Add to overall counts
            total_overall += total
            correct_1_overall += correct_1
            correct_k_overall += correct_k
            correct_expert_overall += correct_expert
            correct_assign_overall += correct_assign

            # Task-wise accuracies
            accuracy_1 = correct_1 / total
            accuracy_k = correct_k / total
            accuracy_expert = correct_expert / total
            accuracy_assign = correct_assign / total

            # Summarize task-wise accuracies
            writer.add_scalar(
                'accuracy_1/%s/%s/%s' % (eval_title, self.name, task_index),
                accuracy_1, step
            )
            writer.add_scalar(
                'accuracy_%s/%s/%s/%s' %
                (k, eval_title, self.name, task_index), accuracy_k, step
            )
            writer.add_scalar(
                'accuracy_expert/%s/%s/%s' %
                (eval_title, self.name, task_index), accuracy_expert, step
            )
            writer.add_scalar(
                'accuracy_assign/%s/%s/%s' %
                (eval_title, self.name, task_index), accuracy_assign, step
            )

        # Overall accuracies
        accuracy_1 = correct_1_overall / total_overall
        accuracy_k = correct_k_overall / total_overall
        accuracy_expert = correct_expert_overall / total_overall
        accuracy_assign = correct_assign_overall / total_overall

        # Summarize overall accuracies
        writer.add_scalar(
            'accuracy_1/%s/%s/overall' % (eval_title, self.name),
            accuracy_1, step
        )
        writer.add_scalar(
            'accuracy_%s/%s/%s/overall' % (k, eval_title, self.name),
            accuracy_k, step
        )
        writer.add_scalar(
            'accuracy_expert/%s/%s/overall' %
            (eval_title, self.name), accuracy_expert, step
        )
        writer.add_scalar(
            'accuracy_assign/%s/%s/overall' %
            (eval_title, self.name), accuracy_assign, step
        )

    def _eval_discriminative_model(
            self,
            model,
            writer: SummaryWriter,
            step, eval_title):
        training = model.training
        model.eval()

        K = 5
        totals = []
        corrects_1 = []
        corrects_k = []

        # Accuracy of each subset
        for subset_name, subset in self.subsets.items():
            data = DataLoader(
                subset,
                batch_size=self.config['eval_batch_size'],
                num_workers=self.config['eval_num_workers'],
                collate_fn=self.collate_fn,
            )
            total = 0.
            correct_1 = 0.
            correct_k = 0.

            for x, y in iter(data):
                b = x.size(0)
                with torch.no_grad():
                    logits = model(x).view(b, -1)
                # [B, K]
                _, pred_topk = logits.topk(K, dim=1)
                correct_topk = (
                        pred_topk.cpu() == y.view(b, -1).expand_as(pred_topk)
                ).float()
                correct_1 += correct_topk[:, :1].view(-1).cpu().sum()
                correct_k += correct_topk[:, :K].view(-1).cpu().sum()
                total += x.size(0)
            totals.append(total)
            corrects_1.append(correct_1)
            corrects_k.append(correct_k)
            accuracy_1 = correct_1 / total
            accuracy_k = correct_k / total
            writer.add_scalar(
                'accuracy_1/%s/%s/%s' % (eval_title, self.name, subset_name),
                accuracy_1, step
            )
            writer.add_scalar(
                'accuracy_%d/%s/%s/%s' %
                (K, eval_title, self.name, subset_name), accuracy_k, step
            )

        # Overall accuracy
        total = sum(totals)
        correct_1 = sum(corrects_1)
        correct_k = sum(corrects_k)
        accuracy_1 = correct_1 / total
        accuracy_k = correct_k / total
        writer.add_scalar(
            'accuracy_1/%s/%s/overall' % (eval_title, self.name),
            accuracy_1, step
        )
        writer.add_scalar(
            'accuracy_%d/%s/%s/overall' % (K, eval_title, self.name),
            accuracy_k, step
        )
        model.train(training)

    def _eval_generative_model(
            self,
            model,
            writer: SummaryWriter,
            step, eval_title):
        # change the model to eval mode
        training = model.training
        z_samples = model.config['z_samples']
        model.eval()
        model.config['z_samples'] = 16
        # evaluate generative model on each subset
        subset_counts = []
        subset_cumulative_bpds = []
        for subset_name, subset in self.subsets.items():
            data = DataLoader(
                subset,
                batch_size=self.config['eval_batch_size'],
                num_workers=self.config['eval_num_workers'],
                collate_fn=self.collate_fn,
            )
            subset_count = 0
            subset_cumulative_bpd = 0
            # evaluate on a subset
            for x, _ in iter(data):
                dim = reduce(lambda x, y: x * y, x.size()[1:])
                with torch.no_grad():
                    ll = model(x)
                bpd = -ll / math.log(2) / dim
                subset_count += x.size(0)
                subset_cumulative_bpd += bpd.sum()
            # append the subset evaluation result
            subset_counts.append(subset_count)
            subset_cumulative_bpds.append(subset_cumulative_bpd)
            subset_bpd = subset_cumulative_bpd / subset_count
            writer.add_scalar(
                'bpd/%s/%s/%s' % (eval_title, self.name, subset_name),
                subset_bpd, step
            )
        # Overall accuracy
        overall_bpd = sum(subset_cumulative_bpds) / sum(subset_counts)
        writer.add_scalar(
            'bpd/%s/%s/overall' % (eval_title, self.name), overall_bpd, step
        )
        # roll back the mode
        model.train(training)
        model.config['z_samples'] = z_samples

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

    def __init__(self, config, domain_id, train=True):
        self.domain_id = domain_id

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

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target = super(MNIST, self).__getitem__(index)
        return img, target, self.domain_id


class SVHN(torchvision.datasets.SVHN, ClassificationDataset):
    name = 'svhn'
    num_classes = 10

    def __init__(self, config, domain_id, train=True):
        self.domain_id = domain_id
        # Compose transformation
        transform_list = [
            transforms.Resize((config['x_h'], config['x_w'])),
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

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target = super(SVHN, self).__getitem__(index)
        return img, target, self.domain_id


class CIFAR10(torchvision.datasets.CIFAR10, ClassificationDataset):
    name = 'cifar10'
    num_classes = 10

    def __init__(self, config, domain_id, train=True):
        self.domain_id = domain_id
        if config.get('augment_cifar'):
            transform = transforms.Compose([
                transforms.Resize((config['x_h'], config['x_w'])),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((config['x_h'], config['x_w'])),
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

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target = super(CIFAR10, self).__getitem__(index)
        return img, target, self.domain_id


class USPS(torchvision.datasets.USPS, ClassificationDataset):
    name = 'usps'
    num_classes = 10

    def __init__(self, config, domain_id, train=True):
        self.domain_id = domain_id
        if config.get('augment_usps'):
            transform = transforms.Compose([
                transforms.Resize((config['x_h'], config['x_w'])),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((config['x_h'], config['x_w'])),
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

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target = super(USPS, self).__getitem__(index)
        return img, target, self.domain_id


class CIFAR100(torchvision.datasets.CIFAR100, ClassificationDataset):
    name = 'cifar100'
    num_classes = 100

    def __init__(self, config, domain_id, train=True):
        self.domain_id = domain_id
        if config.get('augment_cifar'):
            transform = transforms.Compose([
                transforms.Resize((config['x_h'], config['x_w'])),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((config['x_h'], config['x_w'])),
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
