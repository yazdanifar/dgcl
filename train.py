import os
import pickle
import torch
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from data import DataScheduler
import torch.optim as optim

from models.model_diva import DIVA


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
    class_num = config['diva']['y_dim']
    domain_num = config['diva']['d_dim']

    device = config['device']
    train_loss = 0
    epoch_class_y_loss = 0
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=config['diva']['lr'])

    for step, (x, y, d, t) in enumerate(scheduler):
        step += 1
        print('\r[Step {:4}]'.format(step), end='')

        summarize = step % config['summary_step'] == 0
        summarize_samples = summarize and config['summarize_samples']

        y_eye = torch.eye(class_num)
        y = y_eye[y]

        # Convert to onehot
        d_eye = torch.eye(domain_num)
        d = d_eye[d]

        # learn the model
        x, y, d = x.to(device), y.to(device), d.to(device)

        if summarize_samples and (step == 1): # TODO: need attention
            save_reconstructions(model, d, x, y, writer, t)

        optimizer.zero_grad()
        loss, class_y_loss = model.loss_function(d, x, y)
        loss.backward()
        optimizer.step()

        train_loss += loss
        epoch_class_y_loss += class_y_loss

        # Evaluate the model

        if step % config['eval_step'] == 0:
            scheduler.eval(model, model.classifier, writer, step, 'model')

    train_loss /= len(scheduler)
    epoch_class_y_loss /= len(scheduler)


def save_reconstructions(model, d, x, y, writer, step):
    # Save reconstuction
    with torch.no_grad():
        x_recon, _, _, _, _, _, _, _, _, _, _, _ = model.forward(d, x, y)
        recon_batch = x_recon.view(-1, 1, 28, 28, 256)

        sample = torch.zeros(100, 1, 28, 28).cuda()

        for i in range(28):
            for j in range(28):

                # out[:, :, i, j]
                probs = F.softmax(recon_batch[:, :, i, j], dim=2).data

                # Sample single pixel (each channel independently)
                for k in range(1):
                    # 0 ~ 255 => 0 ~ 1
                    val, ind = torch.max(probs[:, k], dim=1)
                    sample[:, k, i, j] = ind.squeeze().float() / 255.

        n = min(x.size(0), 8)
        comparison = torch.cat([x.view(100, 1, 28, 28)[:n],
                                sample[:n]])

        # collage = _make_collage(samples, config, grid_h, grid_w)
        writer.add_image('samples/{}'.format(i + 1), comparison, step)  # need check

        # save_image(comparison.cpu(),
        #            'reconstruction_only_sup_' + str(epoch) + '.png', nrow=n)
