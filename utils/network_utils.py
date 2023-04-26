import torch
import os
import time

from torchvision.utils import flow_to_image, make_grid
import torch.nn.functional as F
import cv2

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def gridify(args, seq, outputs, frame, batch_idx):
    seq_cpu = {}
    seq_cpu['image'] = [x.to(args.device) for x in seq['image']]
    seq_cpu['flow'] = [x.to(args.device) for x in seq['flow']]

    a = seq_cpu['image'][frame][batch_idx]*255.0 # input t
    d = seq_cpu['image'][frame-1][batch_idx]*255.0
    h = flow_to_image(outputs['flow'][batch_idx])
    i = flow_to_image(seq_cpu['flow'][frame][batch_idx])
    grid = make_grid([a,d,h,i], nrow=2, padding=20)
    grid = F.interpolate(grid.unsqueeze(0), scale_factor=.35).squeeze(0)

    return cv2.cvtColor(grid.permute(1,2,0).cpu().numpy(), cv2.COLOR_BGR2RGB)

def warp_flow(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)

    output = F.grid_sample(x, vgrid, align_corners=True)

    return output

def model_save(model, optimizer, scheduler, epoch, args, save_best=False):

    save_dict = {
        'state': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'args': args,
        'epoch': epoch,
    }
    torch.save(save_dict, os.path.join(args.out, 'models/ckpt.pth'))

    if save_best:
        torch.save(save_dict, os.path.join(args.out, 'models/best_ckpt.pth'))

    if epoch % args.save_every == 0:
        torch.save(save_dict, os.path.join(args.out, 'models/ckpt_{}.pth'.format(epoch)))


def model_load(path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state'], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'], strict=False)
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'], strict=False)
    epoch = checkpoint['epoch']

    return epoch


def measure_efficiency(args):
    import sys
    sys.path.append('core')
    from raft import RAFT
    from argparse import ArgumentParser
    from fvcore.nn import FlopCountAnalysis

    parser = ArgumentParser(description='Parser of Training Arguments')

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
    parser.add_argument('--max_flow', dest='max_flow', help='Set magnitude of flows to exclude from loss', default=150, type=int)
    parser.add_argument('--prev_frames', dest='prev_frames', help='Set number of previous frames', default=1, type=int)
    parser.add_argument("--device", dest='device', default="cuda", type=str)



    args = parser.parse_args()

    model = RAFT(args, ['flow']).to(args.device)
    dims = 800, 800
    x1, x2 = [torch.randn((1, 3, *dims)).cuda(non_blocking=True), torch.randn((1, 3, *dims)).cuda(non_blocking=True)]
    times = []
    with torch.no_grad():
        for i in range(60):
            torch.cuda.synchronize()
            test_time_start = time.time()
            _ = model.forward(x1, x2)
            torch.cuda.synchronize()
            times.append(time.time() - test_time_start)

    fps = round(1 / (sum(times[30:])/len(times[30:])), 2)
    params = count_parameters(model) / 10 ** 6
    flops = FlopCountAnalysis(model, (x1, x2)).total() * 1e-9
    del model
    return params, fps, flops
