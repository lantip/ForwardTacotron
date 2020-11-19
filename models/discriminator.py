import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.modules.rnn import LSTM

from models.tacotron import CBHG
import torch.nn as nn


class Discriminator(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(1, 256, kernel_size=3, stride=1, padding=2)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
        ])
        self.lstm = LSTM(256, 128, batch_first=True, bidirectional=True)
        self.lin = Linear(256, 1)

    def forward(self, x):
        #x = x.transpose(1, 2)
        features = list()
        for module in self.convs:
            x = module(x)
            features.append(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        features.append(x.transpose(1, 2))
        x = self.lin(x)
        features.append(x.transpose(1, 2))
        return features[:-1], features[-1]


if __name__ == '__main__':
    model = Discriminator()

    x = torch.randn(3, 80, 100)
    print(x.shape)

    out = model(x)
    out2 = model(x)
    for (feats_fake, score_fake), (feats_real, _) in zip(out, out2):
        print(feats_fake.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)