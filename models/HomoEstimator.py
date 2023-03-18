import torch
import torch.nn as nn

class HomoEstimator4(nn.Module):
    def __init__(self):
        super(HomoEstimator4, self).__init__()
        self.conv2 = torch.nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(inplace=True))

        self.conv3 = torch.nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(inplace=True))

        self.conv4 = torch.nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(inplace=True))


        self.head = torch.nn.Sequential(nn.Conv2d(256, 8, kernel_size=3, padding=1))


    def forward(self, m1, m2):
        x = self.conv2(torch.cat([m1, m2], 1)) # 260 x 80 x 106 -> 256 x 40 x 53
        x = self.conv3(x)  # 256 x 40 x 53 -> 256 x 20 x 26
        x = self.conv4(x) # 256 x 20 x 26 -> 256 x 10 x 13
        x = self.head(x)  # 256 x 20 x 26 -> 1 x 8 x 1
        return x

class HomoEstimator2(nn.Module):
    def __init__(self):
        super(HomoEstimator2, self).__init__()

        self.conv1 = torch.nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=True))

        self.conv2 = torch.nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=True))

        self.conv3 = torch.nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=True))

        self.conv4 = torch.nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=True))

        self.head = torch.nn.Sequential(nn.Conv2d(128, 2, kernel_size=3, padding=1))


    def forward(self, m1, m2):
        x = self.conv1(torch.cat([m1, m2], 1)) # 64 x 80 x 104 -> 128 x 40 x 52
        x = self.conv2(x)  # 128 x 40 x 52 -> 256 x 20 x 26
        x = self.conv3(x) # 256 x 20 x 26 -> 256 x 10 x 13
        x = self.conv4(x)
        x = self.head(x)  # 256 x 10 x 13 -> 1 x 8 x 1
        return x

class HomoEstimator(nn.Module):
    def __init__(self):
        super(HomoEstimator, self).__init__()

        self.conv1 = torch.nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))

        self.conv2 =  torch.nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))

        self.conv3 =  torch.nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))

        self.conv4 =  torch.nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))


        self.head = torch.nn.Sequential(nn.Conv2d(64, 2, kernel_size=3, padding=1))

    def forward(self, m1, m2):
        x = self.conv1(torch.cat([m1, m2], 1)) # 32 x 320 x 416 -> 64 x 160 x 208
        x = self.conv2(x) # 64 x 160 x 208 -> 128 x 40 x 52
        x = self.conv3(x)  # 128 x 40 x 52 -> 128 x 40 x 52
        x = self.conv4(x) # 128 x 40 x 52 -> 256 x 20 x 26
        x = self.head(x)  # 256 x 10 x 13 -> 1 x 8 x 1
        return x



