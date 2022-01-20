import copy
import os
import pickle
import random
import time
import psutil

import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from data import DataScheduler
import torch.optim as optim
from models.our_diva.our_diva import OurDIVA
from sklearn.decomposition import PCA

MODEL = {
    "OurDIVA": OurDIVA
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
        config['y_dim'],
        config['x_h'] * grid_h,
        config['x_w'] * grid_w
    )
    return collage


def train_model(config, model,
                scheduler: DataScheduler,
                writer: SummaryWriter,
                prof: torch.profiler.profile):
    global projection_matrix
    projection_matrix = torch.rand(size=(64, 2), device=config['device'])  # torch.eye(64, device=config['device'])  # #

    class_num = config['y_dim']
    domain_num = config['d_dim']

    device = config['device']
    train_loss = 0
    epoch_class_y_loss = 0
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=config['model']['lr'])

    d_eye = torch.eye(domain_num, device=device, requires_grad=False)
    y_eye = torch.eye(class_num, device=device, requires_grad=False)
    prev_t = None

    sum_loss = 0
    sum_loss_count = 0
    sum_replay_loss = 0
    sum_replay_loss_count = 0
    prev_model = model

    prof.start()

    print_times = config['print_times']
    if print_times:
        end_time_ow = time.time()

    for step, (x, y, d, t) in enumerate(scheduler):
        if config['print_times']:
            start_time_ow = time.time()
            print("Data Load:", round((start_time_ow - end_time_ow) * 100, 3))

        step += 1
        print('\r[Step {:4} of {} ({:2.2%})]'.format(step, len(scheduler), step / len(scheduler)), end=' ')
        if step % config['training_loss_step'] == 0:
            p = psutil.Process(os.getpid())
            rss = round(p.memory_info().rss / 1000000000, 9)
            vms = round(p.memory_info().rss / 1000000000, 9)
            writer.add_scalar(
                'memory_prof/%s' % "rss",
                rss, step
            )
            writer.add_scalar(
                'memory_prof/%s' % "vms",
                vms, step
            )

        task_changed = prev_t != t and prev_t is not None
        stage_eval_step = config['eval_step'] if config['eval_step'] is not None else int(
            scheduler.task_step[t] / config['eval_per_task'])
        summarize_step = config['summary_step'] if config['summary_step'] is not None else int(
            scheduler.task_step[t] / config['summary_per_task'])

        model_eval = (step % stage_eval_step == 5 and step > 5) or (
                prev_t is None and config['initial_evaluation']) or (
                             task_changed and config['eval_in_task_change']) or step == len(scheduler) - 1

        summarize = (step % summarize_step == 7 and step > 7)
        summarize_samples = summarize and config['summarize_samples']
        prev_t = t

        if task_changed:
            model.task_learned(t - 1, scheduler)
            prev_model = MODEL[config['model_name']](config, writer, config['device'])
            prev_model.load_state_dict(model.state_dict())
            prev_model.to(config['device'])
            prev_model.eval()

        # Evaluate the model
        if model_eval:
            scheduler.eval(model, model.classifier, writer, step, model.name)
        if summarize_samples:
            print("save reconstructions")
            save_latent_variable(model, scheduler, writer, step)
            save_reconstructions(prev_model, model, scheduler, writer, step, t)

        # to device
        x, d = x.to(device), d.to(device)
        if y is not None:
            y = y.to(device)
        # Train the model
        prof.step()  # after loading data
        # Convert to onehot
        if y is not None:
            y = y_eye[y]
        d = d_eye[d]

        # new task batch
        start_time = time.time()

        optimizer.zero_grad()
        loss, class_y_loss = model.loss_function(d, x, y)
        loss.backward()
        optimizer.step()
        prof.step()  # after training on original data

        with torch.no_grad():
            sum_loss += loss
            sum_loss_count += 1

            train_loss += loss
            epoch_class_y_loss += class_y_loss

        end_time = time.time()
        sum_time = (end_time - start_time)
        if print_times:
            print("training time", round((end_time - start_time) * 100, 2), "SUP" if (class_y_loss != 0) else "UNSUP")

        # replay batches
        if config['replay_ratio'] != 0:
            if random.random() < config['replay_ratio']:
                if print_times:
                    start_time = time.time()

                prof.step()  # start generating data
                generated = prev_model.generate_replay_batch(config['replay_batch_size'])
                prof.step()  # end generating data/start learning from it
                if not generated:
                    continue
                x, y, d = generated
                if print_times:
                    mid_time = time.time()
                optimizer.zero_grad()
                replay_loss, replay_class_y_loss = model.train_with_replayed_data(d_eye[d], x.detach(), y_eye[y])
                if config['equal_loss_scale']:
                    replay_loss *= (loss.detach() / replay_loss.detach()).detach()
                if config['scale_replay_loss_wrt_num_tasks']:
                    replay_loss *= t
                replay_loss *= config['replay_loss_multiplier']
                replay_loss.backward()
                optimizer.step()
                prof.step()  # end learning from generated data
                if print_times:
                    end_time = time.time()
                    sum_time += (end_time - start_time)
                    print("generating time", round((mid_time - start_time) * 100, 3), "train on generated",
                          round((end_time - mid_time) * 100, 3))
                with torch.no_grad():
                    sum_replay_loss += replay_loss
                    sum_replay_loss_count += 1

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

        prof.step()

        if print_times:
            end_time_ow = time.time()
            print("all time:", round((end_time_ow - start_time_ow) * 100, 3), "use full:", round(sum_time * 100, 3))

    prof.stop()
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


def project_to_2d(loc):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(loc.cpu().numpy())
    return principalComponents
    # return torch.matmul(loc, projection_matrix[:loc.size(1)]).cpu().numpy()


def get_latent_variable(model, scheduler: DataScheduler, data_loader: DataLoader, domain_id, subplotnum):
    # use the right data loader
    y_eye = torch.eye(scheduler.class_num, device=scheduler.device)
    d_eye = torch.eye(scheduler.domain_num, device=scheduler.device)

    for step, (x, y, d) in enumerate(data_loader):
        # To device
        x, y, d = x.to(scheduler.device), y.to(scheduler.device), d.to(scheduler.device)

        # Convert to onehot
        d_onehot = d_eye[d]
        y_onehot = y_eye[y]

        _, d_hat, y_hat, (_, _), (zd_p_loc, _), zd_q, (_, _) \
            , zx_q, (_, _), zy_q = model(d_onehot, x)
        zy_p_loc, _ = model.pzy(y_onehot)

        return zy_p_loc, zy_q, zd_p_loc, zd_q, y, d


def save_latent_variable(model, scheduler: DataScheduler, writer: SummaryWriter, step):
    with torch.no_grad():
        subplotnum = 3 + len(scheduler.eval_data_loaders)

        ############set title
        for domain_id in range(len(scheduler.eval_data_loaders)):
            plt.figure(domain_id)
            plt.clf()
            plt.title(f"class latent of domain {domain_id}")
        plt.figure(subplotnum - 3)
        plt.clf()
        plt.title("class latent, color=domain")
        plt.figure(subplotnum - 2)
        plt.clf()
        plt.title("domain latent, color=domain")
        plt.figure(subplotnum - 1)
        plt.clf()
        plt.title("domain latent, color=class")

        ########################################
        all_zy_p, all_zy_q, all_zd_p, all_zd_q, all_y, all_d = [], [], [], [], [], []
        for i, eval_related in enumerate(scheduler.eval_data_loaders):
            eval_data_loader, description, start_from = eval_related
            if scheduler.stage >= start_from:
                zy_p, zy_q, zd_p, zd_q, y, d = get_latent_variable(model, scheduler, eval_data_loader, i, subplotnum)
                all_zy_q.append(zy_q)
                all_zy_p.append(zy_p)
                all_zd_p.append(zd_p)
                all_zd_q.append(zd_q)
                all_y.append(y)
                all_d.append(d)

                plt.figure(i)
                zy = project_to_2d(torch.cat([zy_q, zy_p], dim=0))

                zy_q = zy[:zy.shape[0] // 2]
                zy_p = zy[zy.shape[0] // 2:]

                plt.scatter(x=zy_q[:, 0], y=zy_q[:, 1], c=y.cpu().numpy(), cmap=plt.cm.tab20, vmin=0,
                            vmax=model.y_dim - 1)
                plt.scatter(x=zy_p[:, 0], y=zy_p[:, 1], c=y.cpu().numpy(), s=np.ones(zy_p.shape[0]) * 200,
                            cmap=plt.cm.tab20, vmin=0, vmax=model.y_dim - 1, marker='X', edgecolors='black')

        ######################################
        y = torch.cat(all_y, dim=0).cpu().numpy()
        d = torch.cat(all_d, dim=0).cpu().numpy()

        zy = project_to_2d(torch.cat(all_zy_q + all_zy_p, dim=0))
        zy_q = zy[:zy.shape[0] // 2]
        zy_p = zy[zy.shape[0] // 2:]

        zd = project_to_2d(torch.cat(all_zd_q + all_zd_p, dim=0))
        zd_q = zd[:zd.shape[0] // 2]
        zd_p = zd[zd.shape[0] // 2:]

        plt.figure(subplotnum - 3)
        scatter = plt.scatter(x=zy_q[:, 0], y=zy_q[:, 1], c=d, cmap=plt.cm.tab20, vmin=0,
                              vmax=model.d_dim - 1)
        legend1 = plt.legend(*scatter.legend_elements(), loc="upper right", title="Domains")
        plt.gca().add_artist(legend1)

        plt.figure(subplotnum - 2)
        scatter = plt.scatter(x=zd_q[:, 0], y=zd_q[:, 1], c=d, cmap=plt.cm.tab20, vmin=0,
                              vmax=model.d_dim - 1)
        plt.scatter(x=zd_p[:, 0], y=zd_p[:, 1], c=d, s=np.ones(zd_p.shape[0]) * 200,
                    cmap=plt.cm.tab20, vmin=0, vmax=model.d_dim - 1, marker='X', edgecolors='black')
        legend1 = plt.legend(*scatter.legend_elements(), loc="upper right", title="Domains")
        plt.gca().add_artist(legend1)

        plt.figure(subplotnum - 1)
        plt.scatter(x=zd_q[:, 0], y=zd_q[:, 1], c=y, cmap=plt.cm.tab20, vmin=0,
                    vmax=model.y_dim - 1)

        for domain_id in range(len(scheduler.eval_data_loaders)):
            fig = plt.figure(domain_id)
            writer.add_figure(f"zy_per_domain/{domain_id}", fig, step)

        fig = plt.figure(subplotnum - 3)
        writer.add_figure('zy_domain_color', fig, step)

        fig = plt.figure(subplotnum - 2)
        writer.add_figure('zd_domain_color', fig, step)

        fig = plt.figure(subplotnum - 1)
        writer.add_figure('zd_class_color', fig, step)


def save_reconstructions(prev_model, model, scheduler, writer: SummaryWriter, step, task_number):
    # Save reconstuction
    model.eval()
    with torch.no_grad():
        generated = prev_model.generate_replay_batch(10 * task_number)
        if generated:
            x, y, d = generated
            for i in torch.unique(d):
                img = x[d == i]
                img = img.detach()
                writer.add_images('generated_images_batch/%s_%s' % (prev_model.name, i.item()), img, step)

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
                x = model.generate_supervised_image(d_n, y).detach()
                writer.add_images('generated_images_by_domain/%s/domain_%s' % (model.name, d), x, step)
    model.train()
