import torch
import torch.nn as nn
# from torchvision.models import vgg19, VGG19_Weights
from torchvision.models import vgg16, VGG16_Weights
import config


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:36].eval().to(DEVICE)
        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:31].eval().to(config.DEVICE)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, inp, target):
        vgg_input_features = self.vgg(inp)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size, _, h_x, w_x = x.size()

        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])

        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
