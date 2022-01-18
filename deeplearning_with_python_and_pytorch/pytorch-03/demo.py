# _*_ coding: utf-8 _*_
"""
@Author:   Zongzc
@Describe: 利用神经网络对MNIST进行识别（多分类问题）
"""
import numpy as np
import torch
# 导入 pytorch 内置的 mnist 数据
from torchvision.datasets import mnist
# 导入预处理模块
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# 导入nn及优化器
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 定义一些超参数
train_batch_size = 64
test_batch_size = 128
learning_rate = 0.01
num_epoches = 20
lr = 0.01
momentum = 0.5


class Net(nn.Module):
    """
    使用sequential构建网络，Sequential()函数的功能是将网络的层组合到一起
    """

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv1", torch.nn.Conv2d(3, 32, 3, 1, 1)),
                    ("relu1", torch.nn.ReLU()),
                    ("pool", torch.nn.MaxPool2d(2))
                ]
            ))

        self.dense = torch.nn.Sequential(
            OrderedDict([
                ("dense1", torch.nn.Linear(32 * 3 * 3, 128)),
                ("relu2", torch.nn.ReLU()),
                ("dense2", torch.nn.Linear(128, 10))
            ])
        )


if __name__ == '__main__':
    # 定义预处理函数
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    # 下载数据，并对数据进行预处理
    # 已有数据，download=False
    train_dataset = mnist.MNIST('./data', train=True, transform=transform, download=False)
    test_dataset = mnist.MNIST('./data', train=False, transform=transform)
    # 得到一个生成器
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # 可视化源数据
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # 检测是否有可用的GPU，有则使用，否则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs")
    model = Net(28 * 28, 300, 100, 10)
    model.to(device)

    # 定义损失函数和优化器
    # 交叉熵
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # 开始训练
    losses = []
    acces = []
    eval_losses = []
    eval_acces = []
    writer = SummaryWriter(log_dir='logs', comment='train-loss')

    for epoch in range(2):
        train_loss = 0
        train_acc = 0
        model.train()
        # 动态修改参数学习率
        if epoch % 5 == 0:
            optimizer.param_groups[0]['lr'] *= 0.1
            print(optimizer.param_groups[0]['lr'])
        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)
            # 原来的 img.shape 为 torch.Size([64, 1, 28, 28])
            # 变换后 img.shape 为torch.Size([64, 784])
            # view(img.size(0),-1)会把除了第0维的其他维度变成1维
            # view(-1)会把全部维度变成1维
            # view(img.size(0),img.size(1),-1)会把除了第0,1维的其他维度变为1维
            img = img.view(img.size(0), -1)
            # 前向传播
            out = model(img)
            loss = criterion(out, label)
            # 清空梯度
            # 缺省情况梯度是累加的，在梯度反向传播前，先需把梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            # 基于当前梯度（存储在参数的._grad属性中）更新参数
            optimizer.step()
            # 记录误差
            # loss.item(): Tensor -> number
            train_loss += loss.item()
            # 保存loss的数据与epoch数值
            # 在一个图表中记录一个标量的变化
            writer.add_scalar('Train', train_loss / len(train_loader), epoch)
            # 计算分类的准确率
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            train_acc += acc

        losses.append(train_loss / len(train_loader))
        acces.append(train_acc / len(train_loader))
        # 在测试集上检验效果
        eval_loss = 0
        eval_acc = 0
        # 将模型改为预测模式
        model.eval()
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device)
            img = img.view(img.size(0), -1)
            out = model(img)
            loss = criterion(out, label)
            # 记录误差
            eval_loss += loss.item()
            # 记录准确率
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            eval_acc += acc

        eval_losses.append(eval_loss / len(test_loader))
        eval_acces.append(eval_acc / len(test_loader))
        print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
              .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader),
                      eval_loss / len(test_loader), eval_acc / len(test_loader)))

    plt.title('train loss')
    plt.plot(np.arange(len(losses)), losses)
    plt.plot(np.arange(len(eval_losses)), eval_losses)
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    # plt.legend(['Train Loss'], loc='upper right')
    plt.show()
