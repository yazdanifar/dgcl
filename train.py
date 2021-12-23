import copy
import os
import pickle
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from data import DataScheduler
import torch.optim as optim
from models.model_diva import DIVA, OurDIVA

MODEL = {
    "DIVA": DIVA,
    "our_DIVA": OurDIVA
    # "ndpm_model": NdpmModel
    # "our": OUR,
}


# import other models


def _write_summary(summary, writer: SummaryWriter, step):
    for summary_type, summary_dict in summary.items():
        if summary_type == 'scalar':
            write_fn = writer.add_scalar
        elif summary_type == 'image':
            write_fn = writer.add_image
        elif summary_type == 'histogram':
            write_fn = writer.add_histogram
        else:
            raise RuntimeError('Unsupported summary type: %s' % summary_type)

        for tag, value in summary_dict.items():
            write_fn(tag, value, step)


def _make_collage(samples, config, grid_h, grid_w):
    s = samples.view(
        grid_h, grid_w,
        config['x_c'], config['x_h'], config['x_w']
    )
    collage = s.permute(2, 0, 3, 1, 4).contiguous().view(
        config['x_c'],
        config['x_h'] * grid_h,
        config['x_w'] * grid_w
    )
    return collage


def train_model(config, model: DIVA,
                scheduler: DataScheduler,
                writer: SummaryWriter):
    class_num = config['DIVA']['y_dim']
    domain_num = config['DIVA']['d_dim']

    device = config['device']
    train_loss = 0
    epoch_class_y_loss = 0
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=config['DIVA']['lr'])

    d_eye = torch.eye(domain_num)
    y_eye = torch.eye(class_num)
    prev_t = None

    sum_loss = 0
    sum_loss_count = 0
    sum_replay_loss = 0
    sum_replay_loss_count = 0
    prev_model = model

    for step, (x, y, d, t) in enumerate(scheduler):
        step += 1
        print('\r[Step {:4}]'.format(step), end=' ')

        change_task = prev_t != t and prev_t is not None
        stage_eval_step = config['eval_step'] if config['eval_per_task'] is None else int(
            scheduler.task_step[t] / config['eval_per_task'])
        model_eval = step % stage_eval_step == 0 or (prev_t is None and config['initial_evaluation']) or (
                change_task and config['eval_in_task_change']) or step == len(scheduler) - 1
        summarize = step % config['summary_step'] == 0 or 1200 <= step <= 1210
        summarize_samples = summarize and config['summarize_samples']
        prev_t = t

        if change_task:
            scheduler.learn_task(t - 1)
            prev_model = MODEL[config['model_name']](config['DIVA'], config['batch_size'], writer, config['device'])
            prev_model.load_state_dict(model.state_dict())
            prev_model.to(config['device'])
            prev_model.eval()

        # Evaluate the model
        if model_eval:
            scheduler.eval(model, model.classifier, writer, step, model.name)
        if summarize_samples:
            print("save reconstructions", end=" ")
            save_reconstructions(prev_model, model, scheduler, writer, step)

        # Train the model

        # Convert to onehot
        if y is not None:
            y = y_eye[y].to(device)
        d = d_eye[d]

        x, d = x.to(device), d.to(device)

        # new task batch
        optimizer.zero_grad()
        loss, class_y_loss = model.loss_function(d, x, y)
        loss.backward()
        optimizer.step()

        # replay batches
        if config['replay_ratio'] != 0:
            for domain_id in range(domain_num):
                if random.random() < config['replay_ratio'] and len(scheduler.learned_class[domain_id]) > 0:
                    x, y, d = prev_model.get_replay_batch(scheduler.learned_class[domain_id], config['replay_batch_size'])
                    if y is not None:
                        y = y_eye[y].to(device)
                    d = d_eye[d]
                    x, d = x.to(device), d.to(device)

                    optimizer.zero_grad()
                    replay_loss, class_y_loss = model.loss_function(d, x, y)
                    if config['equal_loss_scale']:
                        replay_loss *= (loss.detach() / replay_loss.detach()).detach()
                    replay_loss *= config['replay_loss_multiplier']
                    replay_loss.backward()
                    optimizer.step()

                    sum_replay_loss += replay_loss
                    sum_replay_loss_count += 1

        sum_loss += loss
        sum_loss_count += 1
        if step % config['training_loss_step'] == 0:
            if sum_loss_count != 0:
                writer.add_scalar(
                    'training_loss/%s_%s' % (model.name, "normal"),
                    sum_loss / sum_loss_count, step
                )
            if sum_replay_loss_count != 0:
                writer.add_scalar(
                    'training_loss/%s_%s' % (model.name, "replay"),
                    sum_replay_loss / sum_replay_loss_count, step
                )
            sum_loss = 0
            sum_loss_count = 0

            sum_replay_loss = 0
            sum_replay_loss_count = 0
        train_loss += loss
        epoch_class_y_loss += class_y_loss

    train_loss /= len(scheduler)
    epoch_class_y_loss /= len(scheduler)


def show_batch(dd, xx, y, t):
    dd, xx = dd.cpu(), xx.cpu()
    print(f"d shape: {dd.shape} x shape:{xx.shape}")
    for i in range(min(1, xx.shape[0])):
        d, x = dd[i], xx[i]
        plt.imshow(x.permute(1, 2, 0), cmap='gray')
        if y is None:
            print("X", x.shape, "Y is None", "d", d.shape, "task id", t)
            plt.title(f" y = None   d={d}   task:{t}")
        else:
            print("X", x.shape, "Y", y.shape, "d", d.shape, "task id", t)
            plt.title(f" y = {y[i]}   d={d}  task:{t}")

        plt.show()


def save_reconstructions(prev_model: DIVA, model: DIVA, scheduler, writer: SummaryWriter, step):
    # Save reconstuction
    model.eval()
    for i in range(model.d_dim):
        if len(scheduler.learned_class[i]) > 0:
            x, y, d = prev_model.get_replay_batch(scheduler.learned_class[i], 10)
            writer.add_images('generated_images_batch/%s_%s' % (prev_model.name, i), x, step)

    with torch.no_grad():
        all_classes = []
        for i in range(scheduler.stage + 1):
            all_classes += scheduler.stage_classes(i)

        for d in range(model.d_dim):
            dd = []
            yy = []
            for y in range(model.y_dim):
                if (d, y) in all_classes:
                    yy.append(y)
                    dd.append(d)
            if len(yy) > 0:
                y, d_n = np.array(yy).astype(int), np.array(dd).astype(int)
                x = model.generate_supervised_image(d_n, y)
                writer.add_images('generated_images_by_domain/%s/domain_%s' % (model.name, d), x, step)
    model.train()
