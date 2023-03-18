from models.MIMOUNet.layers import *
from models.HomoEstimator import HomoEstimator, HomoEstimator2, HomoEstimator4
from utils.network_utils import homo2offsets, offsets2homo
import torch.nn.functional as F


class ContractingBlock(nn.Module):
    def __init__(self, block, nr_blocks=5):
        super(ContractingBlock, self).__init__()
        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, nr_blocks, name=block),
            EBlock(base_channel*2, nr_blocks, name=block),
            EBlock(base_channel*4, nr_blocks, name=block)
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2)
        ])


    def forward(self, x1):
        x2 = F.interpolate(x1, scale_factor=0.5)
        x4 = F.interpolate(x2, scale_factor=0.5)

        f1 = self.Encoder[0](self.feat_extract[0](x1))

        z = self.feat_extract[1](f1)
        f2 = self.Encoder[1](z)

        z = self.feat_extract[2](f2)
        f4 = self.Encoder[2](z)

        x = (x4, x2, x1)
        f = (f4, f2, f1)
        return x, f


class ExpandingBlock(nn.Module):
    def __init__(self, block, nr_blocks=6):
        super(ExpandingBlock, self).__init__()
        base_channel = 32

        self.scale_homo = torch.eye(3).to(args.device)
        self.scale_homo[0, 0] = 2
        self.scale_homo[1, 1] = 2

        self.homoEst4 = HomoEstimator4()
        self.homoEst2 = HomoEstimator2()
        self.homoEst = HomoEstimator()

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, 2, name=block),
            DBlock(base_channel * 2, 2, name=block),
            DBlock(base_channel, 2, name=block)])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=3, relu=True, stride=1)])

        # self.FAMS = nn.ModuleList([FAM(base_channel * 4), FAM(base_channel * 2), FAM(base_channel)])
        #
        self.FAMS = nn.ModuleList([GSA(128), GSA(64), GSA(32)])

        self.ConvsOutD = nn.ModuleList(
            [BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
             BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
             BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)])

        self.ConvsOutS = nn.ModuleList(
            [SD(base_channel * 4),
             SD(base_channel * 2 + 2),
             SD(base_channel + 2)])

        self.FAH = nn.ModuleList(
            [FAM_homo(base_channel * 4),
             FAM_homo(base_channel * 2),
             FAM_homo(base_channel)])

        self.feat_extract = nn.ModuleList([
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True)])

    def forward(self, x_curr, f_curr, f_prv, m_prv, d_prv):
        x1_4, x1_2, x1_1 = x_curr
        f1_4, f1_2, f1_1 = f_curr
        f2_4, f2_2, f2_1 = f_prv
        m2_4, m2_2, m2_1 = m_prv
        d2_4, d2_2, d2_1 = d_prv
        outputsD, outputsH, outputsS = list(), list(), list()

        ################################ SCALE 4 ######################################

        ### Deblurring
        F4 = self.FAMS[0](f1_4, f2_4)
        z4 = self.Decoder[0](F4)
        d1_4 = self.ConvsOutD[0](z4) + x1_4
        outputsD.append(d1_4)

        ### Segmentation
        m1_4 = self.ConvsOutS[0](z4)
        outputsS.append(m1_4)

        ### Homography

        off4 = self.homoEst4(self.FAH[0](f1_4, m1_4, d1_4), self.FAH[0](f2_4, m2_4, d2_4))
        outputsH.append(off4)
        homo4 = offsets2homo(off4, 80, 104)
        homo_hat2 = torch.matmul(torch.matmul(self.scale_homo.to(homo4.device), homo4),
                                 torch.inverse(self.scale_homo.to(homo4.device)))

        wf2_2 = f2_2# kornia.warp_perspective(f2_2, torch.inverse(homo_hat2), (160, 208))
        wm2_2 = m2_2# kornia.warp_perspective(m2_2, torch.inverse(homo_hat2), (160, 208))
        wd2_2 = d2_2#kornia.warp_perspective(d2_2, torch.inverse(homo_hat2), (160, 208))

        ################################ SCALE 2 ######################################

        ### Deblurring
        F2 = self.FAMS[1](f1_2, wf2_2)
        z2 = self.feat_extract[0](z4)
        z2 = torch.cat([z2, F2], dim=1)
        z2 = self.Convs[0](z2)
        z2 = self.Decoder[1](z2)
        d1_2 = self.ConvsOutD[1](z2) + x1_2
        outputsD.append(d1_2)

        ### Segmentation
        m1_4_up = F.interpolate(m1_4, scale_factor=2)
        m1_2 = self.ConvsOutS[1](torch.cat([z2, m1_4_up], 1)) #CHANGES HAPPENED HERE
        outputsS.append(m1_2)

        ### Homography
        off2 = self.homoEst2(self.FAH[1](f1_2, m1_2, d1_2), self.FAH[1](wf2_2, wm2_2, wd2_2))
        homo2 = torch.matmul(offsets2homo(off2, 160, 208), homo_hat2)
        off2 = homo2offsets(homo2, 160, 208)
        outputsH.append(off2)
        homo_hat1 = torch.matmul(torch.matmul(self.scale_homo.to(homo2.device), homo2),
                                 torch.inverse(self.scale_homo.to(homo2.device)))
        wf2_1 = f2_1# kornia.warp_perspective(f2_1, torch.inverse(homo_hat1), (320, 416))
        wm2_1 = m2_1#kornia.warp_perspective(m2_1, torch.inverse(homo_hat1), (320, 416))
        wd2_1 = d2_1#kornia.warp_perspective(d2_1, torch.inverse(homo_hat1), (320, 416))

        ################################ SCALE 1 ######################################

        ### Deblurring
        F1 = self.FAMS[2](f1_1, wf2_1)
        z1 = self.feat_extract[1](z2)
        z1 = torch.cat([z1, F1], dim=1)
        z1 = self.Convs[1](z1)
        z1 = self.Decoder[2](z1)
        d1_1 = self.ConvsOutD[2](z1) + x1_1
        outputsD.append(d1_1)

        ### Segmentation
        m1_2_up = F.interpolate(m1_2, scale_factor=2)
        m1_1 = self.ConvsOutS[2](torch.cat([z1, m1_2_up], 1)) #CHANGES HAPPENED HERE
        outputsS.append(m1_1)

        ### Homography
        off1 = self.homoEst(self.FAH[2](f1_1, m1_1, d1_1), self.FAH[2](wf2_1, wm2_1, wd2_1))
        homo1 = torch.matmul(offsets2homo(off1, 320, 416), homo_hat1)
        off1 = homo2offsets(homo1, 320, 416)
        outputsH.append(off1)

        return [outputsS, outputsD, outputsH]

class VideoMIMOUNet_fake(nn.Module):
    def __init__(self, tasks, block, nr_blocks):
        super(VideoMIMOUNet_fake, self).__init__()
        self.tasks = tasks
        self.encoder = ContractingBlock(block, nr_blocks)
        self.decoder = ExpandingBlock(block, nr_blocks)

    def forward(self, x2, x1, m2, d2):
        x1, f1 = self.encoder(x1)
        y1 = self.decoder(x1, f1, f1, m2, d2)
        return y1

    # def forward(self, x2, x1, m2, d2):
    #
    #     x2, f2 = self.encoder(x2)
    #     x1, f1 = self.encoder(x1)
    #     y1 = self.decoder(x1, f1, f2, m2, d2)
    #     return y1





