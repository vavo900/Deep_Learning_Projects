import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix, accuracy
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = FCN()

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'fcn.th')))

    log_train = tb.SummaryWriter('logs/train')
    log_valid = tb.SummaryWriter('logs/valid')

    optmzr = torch.optim.Adam(model.parameters(), weight_decay=0.00001)

    loss_value = torch.nn.CrossEntropyLoss()

    train_data = load_dense_data('dense_data/train')
    valid_data = load_dense_data('dense_data/valid')

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


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)
