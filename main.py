import warnings
# 忽视警告
warnings.filterwarnings('ignore')

import torch
from mobileNet import Net
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from util import processing_data
from util import data_path

tdata, vdata = processing_data(data_path = data_path + "image", height=160, width=160, batch_size=32)

net = Net()
lossF = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

lostlist = []

for i in range(30):
    tdata = list(tdata)
    length = len(tdata)
    for item in range(length):
        optimizer.zero_grad()
        input = tdata[item][0]
        target = tdata[item][1]
        output = net(input)
        lossitem = lossF(output, target)
        lossitem.backward()
        optimizer.step()
        lostlist.append(lossitem.item())
        print("[%d, %d] loss: %.10f"%(i+1, item+1, lossitem.item()))

with torch.no_grad():
    tdata = list(tdata)
    length = len(tdata)
    total = 0
    correct = 0
    for i in range(length):
        x = tdata[i][0]
        y = tdata[i][1]
        output = net(x)
        total += output.size(0)
        _, predicted = torch.max(output.data, 1)
        for item in range(output.size(0)):
            if(predicted[item] == y[item]):
                correct += 1
    print("测试集：\n正确数：%d\n总数：%d\n准确率：%.4f%%" % (correct, total, 100*correct/total))
    print("\n")

    vdata = list(vdata)
    length = len(vdata)
    total = 0
    correct = 0
    for i in range(length):
        x = vdata[i][0]
        y = vdata[i][1]
        output = net(x)
        total += output.size(0)
        _, predicted = torch.max(output.data, 1)
        for item in range(output.size(0)):
            if(predicted[item] == y[item]):
                correct += 1
    print("验证集：\n正确数：%d\n总数：%d\n准确率：%.4f%%" % (correct, total, 100*correct/total))



torch.save(net.state_dict(), 'results/modelV2.pkl')
plt.plot(lostlist, label = "loss")
plt.legend()
plt.show()
