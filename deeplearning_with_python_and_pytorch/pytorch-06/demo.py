# _*_ coding: utf-8 _*_
"""
@Author: Zongzc
@Describe: 6.7 使用现代经典模型提升性能
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from loguru import logger
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 定义一些超参数
EPOCHES = 40
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
logger.info("Preparing data..")
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
logger.info("Building model..")
net = VGG('VGG19').to(device)
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
# optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()

total_params = sum(p.numel() for p in net.parameters())
logger.info('总参数个数:{}'.format(total_params))
total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
logger.info('需训练参数个数:{}'.format(total_trainable_params))


logger.info("Begin train..")
for epoch in range(EPOCHES):

    for img, label in trainloader:
        img, label = img.to(device), label.to(device)
        # 权重参数梯度清零
        optimizer.zero_grad()
        # 正向及反向传播
        out = net(img)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

    correct = 0
    total = 0
    for img, label in testloader:
        img, label = img.to(device), label.to(device)
        out = net(img)
        _, prediction = torch.max(out, 1)  # 按行取最大值
        pre_num = prediction.cpu().numpy()
        total += label.size(0)
        correct += (pre_num == label.cpu().numpy()).sum()

    logger.debug(f"VGG19模型迭代{str(epoch + 1)}次的正确率为：{str(100 * correct / total)}%")
logger.info("Finish train..")
