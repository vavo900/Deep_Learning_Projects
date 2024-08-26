from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)


def train(args):
    model = model_factory[args.model]()

    """
    Your code here

    """

    batch_sz = 128
    num_eps = 125

    loss_val = ClassificationLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001, momentum=0.9)

    data_path = "data/"
    train_data = load_data(data_path + "train", batch_size=batch_sz)

    for ep in range(num_eps):

        for batch, labels in train_data:

            pred_training = model.forward(batch)

            loss_fwd = loss_val.forward(pred_training, labels)

            opt.zero_grad()
            loss_fwd.backward()
            opt.step()

    # raise NotImplementedError('train')

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
