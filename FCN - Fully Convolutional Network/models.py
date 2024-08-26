import torch
import torch.nn.functional as F


class CNNClassifier(torch.nn.Module):

    class Res(torch.nn.Module):

        def __init__(self, in_chan, out_chan, idx):
            super().__init__()
            conv_kern_sz = 5
            conv_strd = 2
            conv_pad = conv_kern_sz // 2
            pool_kern_sz = 5
            pool_strd  = 2
            pool_pad = pool_kern_sz // 2
            lyr = [
                torch.nn.Conv2d(in_chan, out_chan, conv_kern_sz, conv_strd, conv_pad, bias=False),
                torch.nn.BatchNorm2d(out_chan),
                torch.nn.ReLU(inplace=True), ]
            if idx < 2:
                lyr.append(torch.nn.MaxPool2d(pool_kern_sz, pool_strd, pool_pad))
            self.neur_net = torch.nn.Sequential(*lyr)

        def forward(self, x):
            return self.neur_net(x)

    def __init__(self, lyrs=[64, 256, 512], in_chan=3):
        super().__init__()
        """
        Your code here
        Hint: Base this on yours or HW2 master solution if you'd like.
        Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
        """
        conv_kern_sz = 7
        conv_strd = 2
        conv_pad = conv_kern_sz // 2
        pool_kern_sz = 5
        pool_strd = 2
        pool_pad = pool_kern_sz // 2
        lyr = [torch.nn.Conv2d(in_chan, lyrs[0], conv_kern_sz, conv_strd, conv_pad, bias=False),
               torch.nn.BatchNorm2d(lyrs[0]),
               torch.nn.ReLU(inplace=True),
               torch.nn.MaxPool2d(pool_kern_sz, pool_strd, pool_pad)]

        m = lyrs[0]
        for n, o in enumerate(lyrs):
            lyr.append(self.Res(m, o, n))
            m = o

        self.neur_network = torch.nn.Sequential(*lyr)
        self.clsfr = torch.nn.Linear(m, 6)

        # raise NotImplementedError('CNNClassifier.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        return self.clsfr(self.neur_network(x).mean(dim=[2, 3]))
        # raise NotImplementedError('CNNClassifier.forward')


class FCN(torch.nn.Module):

    class Res(torch.nn.Module):

        def __init__(self, in_chan, out_chan, idx):
            super().__init__()
            conv_kern_sz = 5
            conv_strd = 2
            conv_pad = conv_kern_sz // 2
            lyr = [
                torch.nn.Conv2d(in_chan, out_chan, conv_kern_sz, conv_strd, conv_pad, bias=False),
                torch.nn.BatchNorm2d(out_chan),
                torch.nn.ReLU(inplace=True), ]

            self.neur_net = torch.nn.Sequential(*lyr)

        def forward(self, x):
            return self.neur_net(x)

    def __init__(self, lyrs=[64, 256], in_chan=3):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        conv_kern_sz = 9
        conv_strd = 2
        conv_pad = conv_kern_sz // 2
        class_cnt = 5
        lyr = [torch.nn.Conv2d(in_chan, lyrs[0], conv_kern_sz, conv_strd, conv_pad, bias=False),
             torch.nn.BatchNorm2d(lyrs[0]),
             torch.nn.ReLU(inplace=True), ]
        m = lyrs[0]
        for n, o in enumerate(lyrs):
            lyr.append(self.Res(m, o, n))
            m = o

        lyr.append(torch.nn.Conv2d(lyrs[-1], class_cnt, 1, 1))
        lyr.append(torch.nn.BatchNorm2d(class_cnt))

        lyr.append(torch.nn.ConvTranspose2d(5, 5, (9, 9), (2, 2), (3, 3)))
        lyr.append(torch.nn.BatchNorm2d(class_cnt))

        lyr.append(torch.nn.ConvTranspose2d(5, 5, (5, 5), (2, 2), (2, 2)))
        lyr.append(torch.nn.BatchNorm2d(class_cnt))

        lyr.append(torch.nn.ConvTranspose2d(5, 5, (5, 5), (2, 2), (2, 2)))

        self.neur_network = torch.nn.Sequential(*lyr)

        # raise NotImplementedError('FCN.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,6,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        z = self.neur_network(x)
        W = x.shape[3]
        H = x.shape[2]
        z = z[:, :, :H, :W]
        return z

        # raise NotImplementedError('FCN.forward')


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
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
