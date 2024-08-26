
from .models import   load_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
# from .utils import load_data
from . import dense_transforms
from torch.utils.data import Dataset, DataLoader


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path
        self.data = []
        files = glob(path.join(dataset_path, '*.png'))
        for i, f in enumerate(sorted(files)):
            img = Image.open(f)
            img.load()
            self.data.append((img, i))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(*data)
        return data


def load_data(dataset_path, num_workers=1, batch_size=1):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False, drop_last=True)


def visualize(args):
    from os import path

    viz_logger = tb.SummaryWriter(path.join(args.log_dir, 'viz'), flush_secs=1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_model().eval()

    data = load_data(dataset_path=args.data_dir, num_workers=args.num_workers)
    for img, i in data:
        img.to(device)
        pred = model(img)
        print(f'\rAdding image {i[0]}', end='\r')
        log(viz_logger, img, pred, i)


def log(logger, img, pred, global_step):
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)]) / 2
    ax.add_artist(plt.Circle(WH2 * (pred[0].cpu().detach().numpy() + 1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step[0])
    del ax, fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default='.')
    parser.add_argument('--data_dir', default='test_agent')
    parser.add_argument('-w', '--num_workers', type=int, default=1)

    args = parser.parse_args()
    visualize(args)
