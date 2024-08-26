from .models import PuckDetector, save_model
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms


def train(args):
    from os import path
    model = PuckDetector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'puck_detector.th')))

    loss = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6,
                                                           threshold_mode='rel', threshold=0.01, verbose=True)

    transform = dense_transforms.Compose([
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ToTensor(),
    ])

    train_data = load_data(dataset_path=args.data_dir, transform=transform, num_workers=args.num_workers)

    # mean = torch.stack([i.view(3, -1).mean(dim=1) for i, _ in train_data]).mean(dim=0)
    # std = torch.stack([i.view(3, -1).std(dim=1) for i, _ in train_data]).mean(dim=0)
    # print(f'mean: {mean}')
    # print(f'std: {std}')

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        losses = []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            pred = model(img)
            loss_val = loss(pred, label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
                log(train_logger, img, label, pred, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

            losses.append(loss_val.detach().cpu().numpy())

        avg_loss = np.mean(losses)
        print('epoch %-3d \t loss = %0.3f' % (epoch, avg_loss))

        total_loss = np.sum(losses)
        scheduler.step(avg_loss)

        if train_logger is not None:
            train_logger.add_scalar('mean_loss', avg_loss, epoch)
            train_logger.add_scalar('total_loss', total_loss, epoch)

        save_model(model)

    save_model(model)


def log(logger, img, label, pred, global_step):
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)]) / 2
    ax.add_artist(plt.Circle(WH2 * (label[0].cpu().detach().numpy() + 1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2 * (pred[0].cpu().detach().numpy() + 1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default='.')
    parser.add_argument('--data_dir', default='drive_data')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-w', '--num_workers', type=int, default=16)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])')

    args = parser.parse_args()
    train(args)
