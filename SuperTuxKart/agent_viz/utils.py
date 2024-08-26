import json
import numpy as np

from torch.utils.data import Dataset, DataLoader
from . import dense_transforms


DATASET_PATH = 'drive_data'


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path
        self.data = []
        for f in glob(path.join(dataset_path, '*.json')):
            i = Image.open(f.replace('.json', '.png'))
            i.load()
            with open(f) as f_:
                data = json.load(f_)
                puck_location = data.get('puck_img_coords', [0.0, 1.0])
                if not data.get('puck_in_view', True):
                    puck_location = [0.0, 1.0]
            self.data.append((i, np.array(puck_location)))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(*data)
        return data


def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)
