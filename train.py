import os
import pickle
import torch
from tensorboardX import SummaryWriter
from models.model_diva import DIVA
from data import DataScheduler


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
    for step, (x, y, t) in enumerate(scheduler):
        step += 1
        print('\r[Step {:4}]'.format(step), end='')

        summarize = step % config['summary_step'] == 0
        summarize_samples = summarize and config['summarize_samples']

        # learn the model
        model.learn(x, y, t, step)

        # Evaluate the model

        if evaluatable and step % config['eval_step'] == 0:
            scheduler.eval(model, writer, step, 'model')

        # Summarize samples
        if summarize_samples:
            is_ndpm = isinstance(model, NdpmModel)
            comps = [e.g for e in model.ndpm.experts[1:]] \
                if is_ndpm else [model.component]
            if len(comps) == 0:
                continue
            grid_h, grid_w = config['sample_grid']
            total_samples = []
            # Sample from each expert
            for i, expert in enumerate(comps):
                with torch.no_grad():
                    samples = expert.sample(grid_h * grid_w)
                total_samples.append(samples)
                collage = _make_collage(samples, config, grid_h, grid_w)
                writer.add_image('samples/{}'.format(i + 1), collage, step)