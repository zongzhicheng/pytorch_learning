# _*_ coding: utf-8 _*_
"""
@Author:   Zongzc
@Describe: 利用预训练模型dehazer.pth清除图像中的雾霾
"""
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import matplotlib.pyplot as plt
from matplotlib.image import imread


# 定义一个神经网络
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
        self.e_conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
        self.e_conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        concat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.e_conv3(concat1))

        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.e_conv4(concat2))
        concat3 = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.relu(self.e_conv5(concat3))
        clean_image = self.relu((x5 * x) - x5 + 1)
        return clean_image


def cl_image(image_path):
    data = Image.open(image_path)
    data = (np.asarray(data) / 255.0)
    data = torch.from_numpy(data).float()
    data = data.permute(2, 0, 1)
    data = data.to(device).unsqueeze(0)

    net.load_state_dict(torch.load('clean_photo/dehazer.pth'))

    clean_image = net.forward(data)
    torchvision.utils.save_image(torch.cat((data, clean_image), 0),
                                 "clean_photo/results/" + image_path.split("\\")[-1])


if __name__ == '__main__':
    # 创建存放目标文件目录
    path = 'clean_photo/results'
    if not os.path.exists(path):
        os.makedirs(path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Model().to(device)
    print(os.getcwd())
    test_list = glob.glob(os.getcwd() + r"\clean_photo\test_images\*")
    for image in test_list:
        cl_image(image)
        print(image, "done!")

    img = imread('./clean_photo/test_images/shanghai02.jpg')
    plt.imshow(img)
    plt.show()

    img = imread('clean_photo/results/shanghai02.jpg')
    plt.imshow(img)
    plt.show()
