from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

lbl_dict = {'background': 0, 'kart': 1, 'pickup': 2, 'nitro': 3, 'bomb': 4, 'projectile': 5}


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv

        WARNING: Do not perform data normalization here.
        """

        self.lbls = []
        self.imgs = []

        trnsfrmtion = transforms.Compose([transforms.ToTensor()])
        csv_file_path = str(dataset_path) + "/" + "labels.csv"

        with open(csv_file_path) as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            skip_header = 0

            for row in csv_reader:

                if skip_header != 0:

                    file = row[0]
                    img_file_path = str(dataset_path) + "/" + file
                    image = Image.open(str(img_file_path))
                    img_trnsfrm = trnsfrmtion(image)

                    label_str = row[1]
                    label_val = lbl_dict[label_str]
                    self.imgs.append(img_trnsfrm)
                    self.lbls.append(label_val)

                skip_header = 1

        # raise NotImplementedError('SuperTuxDataset.__init__')

    def __len__(self):
        """
        Your code here
        """

        m = len(self.imgs)
        return m

        # raise NotImplementedError('SuperTuxDataset.__len__')

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """

        m = self.imgs[idx], self.lbls[idx]
        return m

        # raise NotImplementedError('SuperTuxDataset.__getitem__')


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
