import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from data import DataScheduler
from sklearn.decomposition import PCA


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


def get_latent_variable_by_data(model, x, y, d, scheduler):
    y_eye = torch.eye(scheduler.class_num, device=scheduler.device)
    d_eye = torch.eye(scheduler.domain_num, device=scheduler.device)

    x, y, d = x.to(scheduler.device), y.to(scheduler.device), d.to(scheduler.device)

    # Convert to onehot
    d_onehot = d_eye[d]
    y_onehot = y_eye[y]

    _, d_hat, y_hat, (_, _), (zd_p_loc, _), zd_q, (_, _) \
        , zx_q, (_, _), zy_q = model(d_onehot, x)
    zy_p_loc, _ = model.pzy(y_onehot)

    return zy_p_loc, zy_q, zd_p_loc, zd_q, y, d


def get_latent_variable(model, scheduler: DataScheduler, data_loader: DataLoader):
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


def save_latent_variable(prev_model, model, scheduler: DataScheduler, writer: SummaryWriter, step):
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
                zy_p, zy_q, zd_p, zd_q, y, d = get_latent_variable(model, scheduler, eval_data_loader)
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

        #########################
        # add generated images latent
        generated = prev_model.generate_replay_batch(100)
        gen_size = 0
        if generated:
            x, y, d = generated
            zy_p, zy_q, zd_p, zd_q, y, d = get_latent_variable_by_data(prev_model, x, y, d, scheduler)
            gen_size = d.shape[0]
            all_zy_q.append(zy_q)
            all_zy_p.append(zy_p)
            all_zd_p.append(zd_p)
            all_zd_q.append(zd_q)
            all_y.append(y)
            all_d.append(d)
        #####################

        y = torch.cat(all_y, dim=0).cpu().numpy()
        d = torch.cat(all_d, dim=0).cpu().numpy()
        zy = project_to_2d(torch.cat(all_zy_q + all_zy_p, dim=0))
        zd = project_to_2d(torch.cat(all_zd_q + all_zd_p, dim=0))

        if gen_size > 0:
            zy_q = zy[:(zy.shape[0] // 2) - gen_size]
            zy_p = zy[zy.shape[0] // 2:-gen_size]
            zd_q = zd[:zd.shape[0] // 2 - gen_size]
            gen_zd_q = zd[zd.shape[0] // 2 - gen_size: zd.shape[0] // 2]
            zd_p = zd[zd.shape[0] // 2:-gen_size]
            gen_d = d[-gen_size:]
            org_d = d[:-gen_size]

            gen_y = y[-gen_size:]
            org_y= y[:-gen_size]
        else:
            zy_q = zy[:(zy.shape[0] // 2)]
            zy_p = zy[zy.shape[0] // 2:]
            zd_q = zd[:zd.shape[0] // 2]
            gen_zd_q = zd[zd.shape[0] // 2: zd.shape[0] // 2]
            zd_p = zd[zd.shape[0] // 2:]
            org_d = d
            org_y=y

        plt.figure(subplotnum - 3)
        scatter = plt.scatter(x=zy_q[:, 0], y=zy_q[:, 1], c=org_d, cmap=plt.cm.tab20, vmin=0,
                              vmax=model.d_dim)
        legend1 = plt.legend(*scatter.legend_elements(), loc="upper right", title="Domains")
        plt.gca().add_artist(legend1)

        plt.figure(subplotnum - 2)
        scatter = plt.scatter(x=zd_q[:, 0], y=zd_q[:, 1], c=org_d, cmap=plt.cm.tab20, vmin=0,
                              vmax=model.d_dim)
        legend1 = plt.legend(*scatter.legend_elements(), loc="upper right", title="Domains")
        plt.gca().add_artist(legend1)

        #### generated images zd
        if gen_size > 0:
            scatter = plt.scatter(x=gen_zd_q[:, 0], y=gen_zd_q[:, 1], c=gen_d, cmap=plt.cm.tab20,
                                  vmin=0, vmax=model.d_dim, edgecolors='black')
            legend2 = plt.legend(*scatter.legend_elements(), loc="upper left", title="Generated Domains")
            plt.gca().add_artist(legend2)

        plt.scatter(x=zd_p[:, 0], y=zd_p[:, 1], c=org_d, s=np.ones(zd_p.shape[0]) * 200,
                    cmap=plt.cm.tab20, vmin=0, vmax=model.d_dim, marker='X', edgecolors='black')


        plt.figure(subplotnum - 1)
        plt.scatter(x=zd_q[:, 0], y=zd_q[:, 1], c=org_y, cmap=plt.cm.tab20, vmin=0,
                    vmax=model.y_dim - 1)

        ######################
        # write in summary writer
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
