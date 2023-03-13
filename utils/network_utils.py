import torch
import os
import time
import fvcore.nn.flop_count as flop_count
from fvcore.nn import FlopCountAnalysis

import kornia


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())



def homo2offsets(homo, h, w):
    """

    Args:
        homo: Torch.Tensor [B, 3, 3]
        h: Int, height
        w: Int, width

    Returns:
        offsets: [B, 4, 2]
    """
    corners = torch.tensor([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).cuda().float()
    offsets = torch.stack([torch.matmul(h, corners.T)[:2].T for h in homo])
    offsets = offsets - corners.unsqueeze(0)[:, :, :2]
    return offsets


def offsets2homo(offsets, h, w):
    """

    Args:
        offsets: offsets computed by network [B, 8]
        h: height of mask
        w: width of mask

    Returns:
        homography matrix [B, 3, 3]
    """
    corners = torch.tensor([[[0, 0], [w, 0], [0, h], [w, h]]]).cuda().float()
    offsets = offsets.view(offsets.shape[0], 4, 2)
    offsets = corners + offsets
    homo = kornia.geometry.transform.get_perspective_transform(corners.repeat(offsets.shape[0], 1, 1), offsets)
    return homo




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

    model = VideoMIMOUNet_fake(['segment', 'deblur', 'homography'], nr_blocks=args.nr_blocks, block=args.block).cuda()
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
