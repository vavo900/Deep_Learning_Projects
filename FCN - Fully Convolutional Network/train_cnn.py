from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES, accuracy
import torch
import torchvision
import torch.utils.tensorboard as tb



def train(args):
    from os import path
    model = CNNClassifier()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th')))

    log_train = tb.SummaryWriter('logs/train')
    log_valid = tb.SummaryWriter('logs/valid')

    optmzr = torch.optim.Adam(model.parameters(), weight_decay=0.00001)

    loss_value = torch.nn.CrossEntropyLoss()

    train_data = load_data('data/train')
    valid_data = load_data('data/valid')

    step_glob = 0
    for ep in range(25):
        model.train()
        arr_loss = []
        arr_v_acc = []
        arr_acc = []

        for img, lbl in train_data:
            img, lbl = img.to(device), lbl.to(device)

            logit = model(img)

            loss = loss_value(logit, lbl.long())
            acc = accuracy(logit, lbl)

            arr_loss.append(loss.detach().cpu().numpy())
            arr_acc.append(acc.detach().cpu().numpy())

            optmzr.zero_grad()
            loss.backward()
            optmzr.step()
            log_train.add_scalar("Training Loss", loss, global_step=step_glob)

            step_glob = step_glob + 1

        acc_average = sum(arr_acc) / len(arr_acc)
        loss_average = sum(arr_loss) / len(arr_loss)

        log_train.add_scalar("Training Accuracy", acc_average, global_step=step_glob)

        model.eval()

        for img, lbl in valid_data:
            img, lbl = img.to(device), lbl.to(device)
            arr_v_acc.append(accuracy(model(img), lbl).detach().cpu().numpy())
        v_acc_average = sum(arr_v_acc) / len(arr_v_acc)
        log_valid.add_scalar("Val Accuracy", v_acc_average, global_step=step_glob)
        print('ep %-2d accuracy = %0.2f accuracy_valid = %0.2f loss_val = %0.2f' % (ep, acc_average, v_acc_average, loss_average))

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)