import torch
from mobileNet import Net
from util import processing_data
from util import data_path

net = Net()
net.load_state_dict(torch.load('results/modelV1.pkl'))

tdata, vdata = processing_data(data_path = data_path + "image", height=160, width=160, batch_size=32)

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

