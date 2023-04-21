import torch
import torch.nn as nn

class HomoEstimator4(nn.Module):
    def __init__(self):
        super(HomoEstimator4, self).__init__()

        self.conv2 = torch.nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(3, stride=2, padding=1))

        self.conv3 = torch.nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(3, stride=2, padding=1)
                                         )

        self.conv4 = torch.nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(32),
                                         nn.ReLU(inplace=True)
                                         )


        self.head = torch.nn.Sequential(nn.Conv2d(32, 2, kernel_size=3, padding=1, bias=False))


    def forward(self, m1, m2):
        x = self.conv2(torch.cat([m1, m2], 1)) # 260 x 80 x 106 -> 256 x 40 x 53
        x = self.conv3(x)  # 256 x 40 x 53 -> 256 x 20 x 26
        x = self.conv4(x) # 256 x 20 x 26 -> 256 x 10 x 13
        x = self.head(x)  # 256 x 20 x 26 -> 1 x 8 x 1
        x = torch.nn.functional.interpolate(x, (200, 200))
        return x

class HomoEstimator2(nn.Module):
    def __init__(self):
        super(HomoEstimator2, self).__init__()


        self.conv2 = torch.nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(3, stride=2, padding=1))

        self.conv3 = torch.nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(32),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(3, stride=2, padding=1)
                                         )

        self.conv4 = torch.nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(32),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(16),
                                         nn.ReLU(inplace=True)
                                         )

        self.head = torch.nn.Sequential(nn.Conv2d(16, 2, kernel_size=3, padding=1))


    def forward(self, m1, m2):
        x = self.conv2(torch.cat([m1, m2], 1)) # 260 x 80 x 106 -> 256 x 40 x 53
        x = self.conv3(x)  # 256 x 40 x 53 -> 256 x 20 x 26
        x = self.conv4(x) # 256 x 20 x 26 -> 256 x 10 x 13
        x = self.head(x)  # 256 x 20 x 26 -> 1 x 8 x 1
        x = torch.nn.functional.interpolate(x, (400, 400))
        return x

class HomoEstimator(nn.Module):
    def __init__(self):
        super(HomoEstimator, self).__init__()

        self.conv2 = torch.nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(32),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(3, stride=2, padding=1))

        self.conv3 = torch.nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(16),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(8),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(3, stride=2, padding=1)
                                         )

        self.conv4 = torch.nn.Sequential(nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(8),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(8, 4, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(4),
                                         nn.ReLU(inplace=True),
                                         )

        self.head = torch.nn.Sequential(nn.Conv2d(4, 2, kernel_size=3, padding=1))

    def forward(self, m1, m2):
        x = self.conv2(torch.cat([m1, m2], 1)) # 260 x 80 x 106 -> 256 x 40 x 53
        x = self.conv3(x)  # 256 x 40 x 53 -> 256 x 20 x 26
        x = self.conv4(x) # 256 x 20 x 26 -> 256 x 10 x 13
        x = self.head(x)  # 256 x 20 x 26 -> 1 x 8 x 1
        x = torch.nn.functional.interpolate(x, (800, 800))
        return x



