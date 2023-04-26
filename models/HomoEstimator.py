import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )



class HomoEstimator4(nn.Module):
    expansion = 1
    def __init__(self,):
        super(HomoEstimator4,self).__init__()

        self.batchNorm = True

        self.conv3   = conv(self.batchNorm, 256,  256, kernel_size=5, stride=2)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)

        self.deconv4 = deconv(512,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.predict_flow5 = predict_flow(512)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(66)

        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x2, x1):
        x = torch.cat([x2, x1], 1)
        out_conv3 = self.conv3(x)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)

        flow5       = self.predict_flow5(out_conv5)  # 2 x 512 x 25 x 25   ->   2 x 2 x 25 x 25
        flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)  #  2 x 2 x 50 x 50   ->   2 x 2 x 50 x 50
        out_deconv4 = crop_like(self.deconv4(out_conv5), out_conv4)  # 2 x 256 x 50 x 50  ->   2 x 2 x 50 x 50

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        return flow2



class HomoEstimator2(nn.Module):
    expansion = 1
    def __init__(self,):
        super(HomoEstimator2,self).__init__()

        self.batchNorm = True
        self.conv2   = conv(self.batchNorm, 128,  128, stride=2)
        self.conv3   = conv(self.batchNorm, 256,  256, stride=2)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)

        self.deconv4 = deconv(512,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.predict_flow5 = predict_flow(512)
        self.predict_flow4 = predict_flow(770) # 512 + 256 + 2
        self.predict_flow3 = predict_flow(386) # 256 + 128 + 2
        self.predict_flow2 = predict_flow(66)  # 64        + 2

        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x2, x1):
        x = torch.cat([x2, x1], 1)
        out_conv2 = self.conv2(x)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)

        flow5       = self.predict_flow5(out_conv5)  # 2 x 512 x 25 x 25   ->   2 x 2 x 25 x 25
        flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)  #  2 x 2 x 50 x 50   ->   2 x 2 x 50 x 50
        out_deconv4 = crop_like(self.deconv4(out_conv5), out_conv4)  # 2 x 256 x 50 x 50  ->   2 x 2 x 50 x 50

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2       = self.predict_flow2(concat2)
        flow2_up    = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv2(concat3)


        concat1 = torch.cat((out_deconv1,flow2_up),1)
        flow1 = self.predict_flow1(concat1)

        return flow2

class HomoEstimator(nn.Module):
    def __init__(self):
        super(HomoEstimator, self).__init__()

        self.conv2 = torch.nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(3, stride=2, padding=1))

        self.conv3 = torch.nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(32),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(16),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(16),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(3, stride=2, padding=1)
                                         )

        self.conv4 = torch.nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(16),
                                         nn.ReLU(inplace=True)
                                         )

        self.head = torch.nn.Sequential(nn.Conv2d(16, 2, kernel_size=3, padding=1))

    def forward(self, m1, m2):
        x = self.conv2(torch.cat([m1, m2], 1)) # 260 x 80 x 106 -> 256 x 40 x 53
        x = self.conv3(x)  # 256 x 40 x 53 -> 256 x 20 x 26
        x = self.conv4(x) # 256 x 20 x 26 -> 256 x 10 x 13
        x = self.head(x)  # 256 x 20 x 26 -> 1 x 8 x 1
        x = torch.nn.functional.interpolate(x, (800, 800))
        return x



