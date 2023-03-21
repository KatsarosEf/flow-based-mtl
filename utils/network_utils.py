import torch
import os
import time

import kornia
from torchvision.utils import flow_to_image, make_grid
import torch.nn.functional as F
import cv2

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def gridify(args, seq, outputs, d2, frame, batch_idx):
    seq_cpu = {}
    seq_cpu['image'] = [x.to(args.device) for x in seq['image']]
    seq_cpu['deblur'] = [x.to(args.device) for x in seq['deblur']]
    seq_cpu['segment'] = [x.to(args.device) for x in seq['segment']]
    seq_cpu['flow'] = [x.to(args.device) for x in seq['flow']]

    a = seq_cpu['image'][frame][batch_idx]*255.0 # input t
    b = outputs['deblur'][2][batch_idx].clip(0, 1) * 255.0
    c = seq_cpu['deblur'][frame][batch_idx]*255.0
    d = seq_cpu['image'][frame-1][batch_idx]*255.0
    e = torch.argmax(outputs['segment'][2][batch_idx].repeat(3, 1, 1))*255.0
    f = seq_cpu['segment'][frame][batch_idx].repeat(3, 1, 1)*255.0
    g = d2[2][batch_idx].clip(0, 1)*255.0
    h = flow_to_image(outputs['flow'][2][batch_idx])
    i = flow_to_image(seq_cpu['flow'][frame][batch_idx].permute(2, 0, 1))
    grid = make_grid([a,b,c,d,e,f,g,h,i], nrow=3, padding=20)
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
    model.load_state_dict(checkpoint['state'], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'], strict=False)
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'], strict=False)
    epoch = checkpoint['epoch']

    return epoch


def measure_efficiency(args):

    from models.MIMOUNet.MIMOUNet_fake import VideoMIMOUNet_fake

    model = VideoMIMOUNet_fake(['segment', 'deblur', 'homography'], nr_blocks=args.nr_blocks, block=args.block).to(args.device)
    dims = 800, 800
    x1, x2 = [torch.randn((1, 3, *dims)).cuda(non_blocking=True), torch.randn((1, 3, *dims)).cuda(non_blocking=True)]
    m2 = [torch.rand((x1.shape[0], 2, 200, 200), device='cuda'),
          torch.rand((x1.shape[0], 2, 400, 400), device='cuda'),
          torch.rand((x1.shape[0], 2, 800, 800), device='cuda')]
    d2 = [torch.rand((x1.shape[0], 3, 200, 200), device='cuda'),
          torch.rand((x1.shape[0], 3, 400, 400), device='cuda'),
          torch.rand((x1.shape[0], 3, 800, 800), device='cuda')]

    times = []
    with torch.no_grad():
        for i in range(60):
            torch.cuda.synchronize()
            test_time_start = time.time()
            _ = model.forward(x1, x2, m2, d2)
            torch.cuda.synchronize()
            times.append(time.time() - test_time_start)

    fps = round(1 / (sum(times[30:])/len(times[30:])), 2)
    params = count_parameters(model) / 10 ** 6
    flops = FlopCountAnalysis(model, (x1, x2, m2, d2)).total() * 1e-9
    del model
    return params, fps, flops


def warp(image, homography):
    return kornia.geometry.warp_perspective(image, homography, image.shape[-2:])
