import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入 3x160x160 输出 16x156x156
        self.Conv1 = nn.Conv2d(3, 16, 5)
        # 输入 16x39*39 输出 32x35x35
        self.Conv2 = nn.Conv2d(16, 32, 5)
        self.l1 = nn.Linear(32*2*2, 2)

    def forward(self, x):
        print(x)
        x = self.Conv1(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 4)
        x = self.Conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 17)
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        return x

if __name__ == "__main__":
    x = torch.rand(12, 3, 160, 160)
    net = Net()
    y = net(x)
    print(y.size())