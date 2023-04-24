from models.MIMOUNet.layers import *
from models.HomoEstimator import HomoEstimator4
from utils.network_utils import warp_flow
import torch.nn.functional as F

class ContractingBlock(nn.Module):
    def __init__(self, args, block, nr_blocks=2):
        super(ContractingBlock, self).__init__()
        self.args = args
        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, nr_blocks, name=block),
            EBlock(base_channel*2, nr_blocks, name=block),
            EBlock(base_channel*4, nr_blocks, name=block)
        ])

        self.feat_extract = nn.ModuleList([
            nn.Sequential(BasicConv(3, base_channel, kernel_size=7, norm=True, relu=True, stride=1),
                          BasicConv(base_channel, base_channel, kernel_size=3, norm=True, relu=True, stride=1)),

            nn.Sequential(BasicConv(base_channel, base_channel*2, kernel_size=5, norm=True, relu=True, stride=2),
                          BasicConv(base_channel*2, base_channel*2, kernel_size=3, norm=True, relu=True, stride=1)),

            nn.Sequential(BasicConv(base_channel*2, base_channel*4, kernel_size=5, norm=True, relu=True, stride=2),
                          BasicConv(base_channel*4, base_channel*4, kernel_size=3, norm=True, relu=True, stride=1))
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
    def __init__(self, args, block, nr_blocks=2):
        super(ExpandingBlock, self).__init__()
        base_channel = 32

        self.of_est4 = HomoEstimator4()

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, 2, name=block),
            DBlock(base_channel * 2, 2, name=block),
            DBlock(base_channel, 2, name=block)])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=3, relu=True, stride=1)])

        self.FAMS = nn.ModuleList([GSA(128), GSA(64), GSA(32)])

        self.ConvsOutS = nn.ModuleList(
            [SD(base_channel * 4),
             SD(base_channel * 2 + 2),
             SD(base_channel * 1 + 2)])

        self.ConvsOutD = nn.ModuleList(
            [DD(base_channel * 4),
             DD(base_channel * 2),
             DD(base_channel * 1)])

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
        outputsD, outputsS = list(), list()

        ################################ SCALE 4 ######################################

        ### Deblurring
        F4 = self.FAMS[0](f1_4, f2_4)
        z4 = self.Decoder[0](F4)
        d1_4 = self.ConvsOutD[0](z4) + x1_4
        outputsD.append(d1_4)

        ### Segmentation
        m1_4 = self.ConvsOutS[0](z4)
        outputsS.append(m1_4)

        ### Flow
        off4 = self.of_est4(self.FAH[0](f1_4, m1_4, d1_4), self.FAH[0](f2_4, m2_4, d2_4)) # offsets 200x200 feature res 200x200
        off4_up = F.interpolate(off4, scale_factor=2.0) * 2.0
        outputsOF = F.interpolate(off4, (800, 800))*4.0

        wf2_2 = warp_flow(f2_2, off4_up)


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
        m1_2 = self.ConvsOutS[1](torch.cat([z2, m1_4_up], 1)) + m1_4_up
        outputsS.append(m1_2)

        off2_up = F.interpolate(off4, scale_factor=4) * 4.0
        wf2_1 = warp_flow(f2_1, off2_up)

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
        m1_1 = self.ConvsOutS[2](torch.cat([z1, m1_2_up], 1)) + m1_2_up
        outputsS.append(m1_1)



        return [outputsS, outputsD, outputsOF]

class VideoMIMOUNet(nn.Module):
    def __init__(self, args, tasks, block, nr_blocks):
        super(VideoMIMOUNet, self).__init__()
        self.tasks = tasks
        self.encoder = ContractingBlock(args, block, nr_blocks).to(args.device)
        self.decoder = ExpandingBlock(args, block, 2).to(args.device)
        self.args = args

    def forward_inference(self, x2, x1, m2, d2):
        x1, f1 = self.encoder(x1)
        y1 = self.decoder(x1, f1, f1, m2, d2)
        return y1

    def forward(self, x2, x1, m2, d2):

        x2, f2 = self.encoder(x2)
        x1, f1 = self.encoder(x1)
        y1 = self.decoder(x1, f1, f2, m2, d2)
        return y1



if __name__ == "__main__":
    import torch
    import time
    from argparse import ArgumentParser
    device = 'cpu'
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())


    parser = ArgumentParser(description='Parser of Training Arguments')

    # parser.add_argument('--data', dest='data_path', help='Set dataset root_path', default='C:\\Users\\User\\PycharmProjects\\raid\\dblab_ecai', type=str) #/media/efklidis/4TB/ # ../raid/data_ours_new_split
    # parser.add_argument('--out', dest='out', help='Set output path', default='C:\\Users\\User\\PycharmProjects\\raid\\ecai-mtl', type=str)

    parser.add_argument('--data', dest='data_path', help='Set dataset root_path', default='../dblab_ecai', type=str) # # ../raid/data_ours_new_split
    parser.add_argument('--out', dest='out', help='Set output path', default='./debug-ecai-mtl', type=str)

    parser.add_argument('--block', dest='block', help='Type of block "fft", "res", "inverted", "inverted_fft" ', default='res', type=str)
    parser.add_argument('--nr_blocks', dest='nr_blocks', help='Number of blocks', default=5, type=int)

    parser.add_argument("--segment", action='store_false', help="Flag for segmentation")
    parser.add_argument("--deblur", action='store_false', help="Flag for  deblurring")
    parser.add_argument("--flow", action='store_false', help="Flag for  homography estimation")

    parser.add_argument('--lr', help='Set learning rate', default=1e-4, type=float)
    parser.add_argument('--wdecay', type=float, default=.0005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=0.8)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--bs', help='Set size of the batch size', default=4, type=int)
    parser.add_argument('--seq_len', dest='seq_len', help='Set length of the sequence', default=5, type=int)
    parser.add_argument('--max_flow', dest='max_flow', help='Set magnitude of flows to exclude from loss', default=100, type=int)
    parser.add_argument('--prev_frames', dest='prev_frames', help='Set number of previous frames', default=1, type=int)
    parser.add_argument("--device", dest='device', default="cpu", type=str)

    parser.add_argument('--epochs', dest='epochs', help='Set number of epochs', default=80, type=int)
    parser.add_argument('--save_every', help='Save model every n epochs', default=1, type=int)
    parser.add_argument("--resume", action='store_true', help="Flag for resume training")
    parser.add_argument('--resume_epoch', dest='resume_epoch', help='Number of epoch to resume', default=0, type=int)
    parser.add_argument('--to_visualize', dest='to_visualize', help='Number of mini seqs to visualize in validation', default=2, type=int)
    args = parser.parse_args()

    model = VideoMIMOUNet(args, ['segment', 'deblur', 'flow'], nr_blocks=5, block='res').to(device)


    dims = 800, 800
    x1, x2 = [torch.randn((1, 3, *dims)).to(device), torch.randn((1, 3, *dims)).to(device)]
    m2 = [torch.rand((x1.shape[0], 2, 200, 200), device=device),
          torch.rand((x1.shape[0], 2, 400, 400), device=device),
          torch.rand((x1.shape[0], 2, 800, 800), device=device)]
    d2 = [torch.rand((x1.shape[0], 3, 200, 200), device=device),
          torch.rand((x1.shape[0], 3, 400, 400), device=device),
          torch.rand((x1.shape[0], 3, 800, 800), device=device)]

    times = []
    with torch.no_grad():
        for i in range(60):
            #torch.cuda.synchronize()
            test_time_start = time.time()
            _ = model.forward_inference(x1, x2, m2, d2)
            #torch.cuda.synchronize()
            times.append(time.time() - test_time_start)

    fps = round(1 / (sum(times[30:])/len(times[30:])), 2)
    params = count_parameters(model) / 10 ** 6
    #flops = FlopCountAnalysis(model, (x1, x2, m2, d2)).total() * 1e-9
    print(fps, params)

