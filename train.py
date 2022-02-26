import os
import random
import time
import psutil

import torch
from tensorboardX import SummaryWriter
from data import DataScheduler
import torch.optim as optim

# from models.our_diva.ClOf import ClOf
import monitoring

MODEL = {
    # "ClOf": ClOf
}


# import other models


def train_model(config, model,
                scheduler: DataScheduler,
                writer: SummaryWriter,
                prof: torch.profiler.profile):
    warmup = config['model']['warm_up']
    print("warm_up", warmup)
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
        stage_lenght = len(scheduler.unsup_dataloader) if scheduler.unsup_dataloader is not None else 0
        stage_lenght += len(scheduler.sup_dataloader) if scheduler.sup_dataloader is not None else 0
        Epoch = step // stage_lenght + 1

        beta_d = min([config['model']['beta_d'], config['model']['beta_d'] * Epoch / warmup])
        beta_y = min([config['model']['beta_y'], config['model']['beta_y'] * Epoch / warmup])
        beta_x = min([config['model']['beta_x'], config['model']['beta_x'] * Epoch / warmup])
        model.beta_d = beta_d
        model.beta_y = beta_y
        model.beta_x = beta_x

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
            monitoring.save_latent_variable(prev_model, model, scheduler, writer, step)
            monitoring.save_reconstructions(prev_model, model, scheduler, writer, step, t)

        # to device
        x, d = x.to(device).float(), d.to(device).long()
        if y is not None:
            y = y.to(device).long()
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
                    'training_loss/%s' % "normal",
                    sum_loss / sum_loss_count, step
                )
            if sum_replay_loss_count != 0:
                writer.add_scalar(
                    'training_loss/%s' % "replay",
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
