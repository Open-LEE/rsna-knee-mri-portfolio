import torch
from torch import nn


class KneeModel(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=False):
        super().__init__()
        if backbone == "tiny":  # CPU integration test only
            self.encoder = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                                         nn.AdaptiveAvgPool2d(1), nn.Flatten())
            features = 16
        else:
            import timm
            self.encoder = timm.create_model(backbone, pretrained=pretrained, in_chans=1, num_classes=0)
            features = self.encoder.num_features
        self.attention = nn.Sequential(nn.Linear(features, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(3 * features + 3, 12))

    def forward(self, x, mask):
        b, planes, slices, h, w = x.shape
        f = self.encoder(x.reshape(b * planes * slices, 1, h, w)).reshape(b, planes, slices, -1)
        weights = self.attention(f).softmax(dim=2)
        f = (f * weights).sum(2) * mask.unsqueeze(-1)
        return self.head(torch.cat([f.flatten(1), mask.float()], dim=1))
