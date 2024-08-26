import torch
import torch.nn.functional as F


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        return F.nll_loss(F.log_softmax(input), target)
        # raise NotImplementedError('ClassificationLoss.forward')


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        in_size = 3 * 64 * 64
        out_size = 6
        m = torch.nn.Linear(in_size, out_size)
        self.lin = m

        # raise NotImplementedError('LinearClassifier.__init__')

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        m = x.size(0)
        l = self.lin(x.view(m, -1))
        return l
        # raise NotImplementedError('LinearClassifier.forward')


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        hid_layer_size = 120
        in_size = 3 * 64 * 64
        out_size = 6

        self.lin_1 = torch.nn.Linear(in_size, hid_layer_size)
        self.lin_2 = torch.nn.Linear(hid_layer_size, out_size)

        self.activ_relu = torch.nn.ReLU()

        # raise NotImplementedError('MLPClassifier.__init__')

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        m = x.size(0)
        l_1 = self.lin_1(x.view(m, -1))
        act = self.activ_relu(l_1)
        l_2 = self.lin_2(act)
        return l_2

        # raise NotImplementedError('MLPClassifier.forward')


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
