from numpy import *
from torch.utils.data import Dataset
# from model import vgg
import torch
# from model2 import resnet18
from torch import nn
import os
import sys
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import cv2
import random
import numpy as np
from model3.model_vgg import Vgg16
from collections import Counter

def testmodel():


    # CBMA  通道注意力机制和空间注意力机制的结合
    class ChannelAttention(nn.Module):
        def __init__(self, in_planes, ratio=16):
            super(ChannelAttention, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化高宽为1
            self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化高宽为1

            # 利用1x1卷积代替全连接
            self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # 平均池化---》1*1卷积层降维----》激活函数----》卷积层升维
            avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
            # 最大池化---》1*1卷积层降维----》激活函数----》卷积层升维
            max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
            out = avg_out + max_out  # 加和操作
            return self.sigmoid(out)  # sigmoid激活操作


    class SpatialAttention(nn.Module):
        def __init__(self, kernel_size=7):
            super(SpatialAttention, self).__init__()

            assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
            padding = kernel_size // 2
            # 经过一个卷积层，输入维度是2，输出维度是1
            self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
            self.sigmoid = nn.Sigmoid()  # sigmoid激活操作

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的平均值  b,1,h,w
            max_out, _ = torch.max(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的最大值  b,1,h,w
            x = torch.cat([avg_out, max_out], dim=1)  # 在第一维度上拼接，变为 b,2,h,w
            x = self.conv1(x)  # 转换为维度，变为 b,1,h,w
            return self.sigmoid(x)  # sigmoid激活操作

    class cbamblock1(nn.Module):
        def __init__(self, channel, ratio=16, kernel_size=7):
            super(cbamblock1, self).__init__()
            self.channelattention = ChannelAttention(channel, ratio=ratio)

        def forward(self, x):
            x = x * self.channelattention(x)  # 将这个权值乘上原输入特征层
            return x

    class cbamblock2(nn.Module):
        def __init__(self, channel, ratio=16, kernel_size=7):
            super(cbamblock2, self).__init__()

            self.spatialattention = SpatialAttention(kernel_size=kernel_size)

        def forward(self, x):
            x = x * self.spatialattention(x)  # 将这个权值乘上原输入特征层
            return x


    class ca(nn.Module):
        def __init__(self, num_class=200):
            super(ca, self).__init__()
            self.backbone1 = Vgg16()
            self.backbone2 = Vgg16()
            palmpth = r'vgg16.pth'
            veinpth = r'vgg16.pth'
            palm_state = torch.load(palmpth)
            for k in list(palm_state.keys()):
                if 'classifer' in k:
                    palm_state.pop(k)
                # elif 'fc' in k:
                #     palm_state.pop(k)

            self.backbone1.load_state_dict(palm_state,strict=False)
            vein_state = torch.load(veinpth)
            for k in list(vein_state.keys()):
                if 'classifer' in k:
                    vein_state.pop(k)
                # elif 'fc' in k:
                #     vein_state.pop(k)
            self.backbone2.load_state_dict(vein_state,strict=False)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.channelattention = cbamblock1(channel=512, ratio=16, kernel_size=7)
            self.spatialattention =cbamblock2(channel=512, ratio=16, kernel_size=7)
            self.f = nn.Linear(512, num_class)
            self.fc = nn.Linear(7*7*512, num_class)
            self.d = nn.Dropout(0.2)
            # self.c1 = nn.Conv2d(1024, 512, kernel_size=1)
            # self.c = nn.Conv2d(512, 290, kernel_size=1)

        def forward(self, x1, x2, x3, x4):
            x1 = self.backbone1(x1)
            x2 = self.backbone2(x2)
            x3 = self.backbone1(x3)
            x4 = self.backbone2(x4)

            B, C, H, W = x1.shape

            x1x = x1.reshape(B, -1)
            x2x = x2.reshape(B, -1)
            x3x = x3.reshape(B, -1)
            x4x = x4.reshape(B, -1)

            x1 = self.pool(x1)
            x2 = self.pool(x2)
            B, C, H, W = x3.shape
            x3 = self.pool(x3)
            x4 = self.pool(x4)
            x = x1 + x2
            x = self.channelattention(x)
            x0 = torch.sigmoid(x)
            x1 = x0 * x1
            x2 = x0 * x2
            x = self.spatialattention(x)
            x01 = torch.sigmoid(x)
            x1 = x01 * x1
            x2 = x01 * x2
            x = x1 + x2

            x = x.reshape(B, -1)
            xx = self.f(x)
            xx = self.d(xx)

            x11 = x1.reshape(B, -1)
            x21 = x2.reshape(B, -1)
            x31 = x3.reshape(B, -1)
            x41 = x4.reshape(B, -1)

            x00=torch.nn.functional.softmax(xx)

            x10 = self.fc(x1x)
            x10 = self.d(x10)
            x10 = torch.nn.functional.softmax(x10)

            x20=self.fc(x2x)
            x20 = self.d(x20)
            x20=torch.nn.functional.softmax(x20)

            x30 = self.fc(x3x)
            x30 = self.d(x30)
            x30 = torch.nn.functional.softmax(x30)

            x40 = self.fc(x4x)
            x40 = self.d(x40)
            x40 = torch.nn.functional.softmax(x40)

            return xx , x, x11, x21, x31, x41, x10, x20, x30, x40, x00

    def trans_form(img):

        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])

        img = transform(img)
        # img = img.unsqueeze(0)
        return img

    a = []

    class TestData(Dataset):
        def __init__(self, print_root_dir, vein_root_dir, training=True):
            self.print_root_dir = print_root_dir
            self.vein_root_dir = vein_root_dir
            self.person_path = os.listdir(self.print_root_dir)

        def __getitem__(self, idx):
            person_name = self.person_path[idx // 3]
            a.append(person_name)
            bb = Counter(a)
            b = bb[person_name] - 1
            print_imgs_path = os.listdir(os.path.join(self.print_root_dir, person_name))
            vein_imgs_path = os.listdir(os.path.join(self.vein_root_dir, person_name))
            length1_imgs = len(print_imgs_path)
            if len(a) == len(print_imgs_path):
                a.clear()
            print_img_path = print_imgs_path[b]
            vein_img_path = vein_imgs_path[b]
            p_img_item_path = os.path.join(self.print_root_dir, person_name, print_img_path)
            v_img_item_path = os.path.join(self.vein_root_dir, person_name, vein_img_path)
            p_img = cv2.imread(p_img_item_path)
            p_img = torch.tensor(p_img / 255.0).to(torch.float).permute(2, 0, 1)
            p_img = trans_form(p_img)
            v_img = cv2.imread(v_img_item_path)
            v_img = torch.tensor(v_img / 255.0).to(torch.float).permute(2, 0, 1)
            v_img = trans_form(v_img)
            return p_img, v_img,  p_img, v_img,person_name,person_name

        def __len__(self):
            return len(self.person_path) * 3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # data_test = TestData('../our/c-Palm-print-test/', '../our/palm-vein-test/')
    data_test = TestData('../casia/print-test-roi/', '../casia/vein-test2/')
    # data_test = TestData('../print-test/', '../vein-test/')
    # data_test = TestData('../casia2/print-train-noqiang/', '../casia2/vein-train2-noqiang/')
    # loader = DataLoader(data_test)
    batch_size = 4
    loader = torch.utils.data.DataLoader(
        data_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)
    print("data_loader = ", loader)
    print("start test......")
    model = ca()

    weights_path = "casia_best_90.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("using {} device.".format(device))

    model.eval()
    accurate = 0
    arr = []
    for epoch in range(1):
        acc = 0.0  # accumulate accurate number / epoch
        num=0
        running_loss = 0.0
        with torch.no_grad():
            bar = tqdm(loader, file=sys.stdout)
            for data_test in bar:
                p_imgs, v_imgs, p2_imgs, v2_imgs, person_name, person2_name = data_test
                p_imgs = p_imgs.to(device)
                v_imgs = v_imgs.to(device)
                p2_imgs = p2_imgs.to(device)
                v2_imgs = v2_imgs.to(device)
                person_labels = [int(_) - 1 for _ in person_name]
                person_labels = torch.tensor(person_labels).to(device)
                person2_labels = [int(_) - 1 for _ in person2_name]
                person2_labels = torch.tensor(person2_labels).to(device)
                outputs, x, x1, x2, x3, x4, x10, x20, x30, x40, x00 = model(p_imgs, v_imgs, p2_imgs, v2_imgs)
                predict_y0 = torch.max(outputs, dim=1)[1]
                person_labels = torch.tensor(person_labels).to(device)
                person2_labels = torch.tensor(person2_labels).to(device)
                acc += torch.eq(predict_y0, person_labels.to(device)).sum().item()
                num = len(loader)*4
        accurate = acc / num
        arr.append(accurate)
        print('[epoch %d] ' % (epoch + 1))
        print('  num:{},test_accuracy:{:.3f},acc:{}'.format(num, accurate, acc))
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

if __name__ == "__main__":
    testmodel()









