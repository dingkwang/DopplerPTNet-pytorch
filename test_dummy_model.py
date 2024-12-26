import torch
import torch.nn as nn
from model.pointtransformer.pointtransformer_seg import DopplerPTNet

class DopplerPTConfig:

    def __init__(self):
        self.num_classes = 7
        self.input_channels = 4
        self.use_xyz = True
        self.device = 'cuda'


def test_model():
    config = DopplerPTConfig()
    model = DopplerPTNet(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        use_xyz=config.use_xyz,
        device=config.device,
    ).to(config.device)

    offset = torch.IntTensor([10000, 20001, 30002]).to(config.device)
    xyz = torch.randn(offset[-1], 3).to(config.device)
    velocity = feat = torch.randn(offset[-1], 1).to(config.device)
    criterion = nn.L1Loss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    output = model(xyz=xyz, feat=feat, offset=offset)
    target = torch.zeros_like(output, dtype=torch.long).to(config.device)
    loss = criterion(output, target)
    print("Loss:", loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    model = test_model()
    print(model)
