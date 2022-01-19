# _*_ coding: utf-8 _*_
"""
@Author: Zongzc
@Describe: 6.7 使用现代经典模型提升性能
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import Counter
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 定义一些超参数
BATCHSIZE = 50
DOWNLOAD_MNIST = False
EPOCHES = 20
LR = 0.001
cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.device('cpu')


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 在笔记本上跑的 别的机器上batch_size可以放大
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

net4 = VGG('VGG16')

mlps = [net4.to(device)]

optimizer = torch.optim.Adam([{"params": mlp.parameters()} for mlp in mlps], lr=LR)

loss_function = nn.CrossEntropyLoss()

for ep in range(EPOCHES):
    for img, label in trainloader:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()  # 10个网络清除梯度
        for mlp in mlps:
            mlp.train()
            out = mlp(img)
            loss = loss_function(out, label)
            loss.backward()  # 网络们获得梯度
        # 对所有的参数进行更新
        optimizer.step()

    pre = []
    vote_correct = 0
    mlps_correct = [0 for i in range(len(mlps))]
    for img, label in testloader:
        img, label = img.to(device), label.to(device)
        for i, mlp in enumerate(mlps):
            mlp.eval()
            out = mlp(img)

            _, prediction = torch.max(out, 1)  # 按行取最大值
            pre_num = prediction.cpu().numpy()
            mlps_correct[i] += (pre_num == label.cpu().numpy()).sum()

            pre.append(pre_num)
        arr = np.array(pre)
        pre.clear()
        result = [Counter(arr[:, i]).most_common(1)[0][0] for i in range(BATCHSIZE)]
        vote_correct += (result == label.cpu().numpy()).sum()
    # print("epoch:" + str(ep) + "集成模型的正确率" + str(vote_correct / len(testloader)))

    for idx, coreect in enumerate(mlps_correct):
        print("VGG16模型迭代" + str(ep) + "次的正确率为：" + str(coreect / len(testloader)))
