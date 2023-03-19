import torch
import os
import time

import kornia
from torchvision.utils import flow_to_image, make_grid



def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def gridify(seq, outputs, d2, frame, batch_idx):

    a = seq['image'][frame][batch_idx]*255.0 # input t
    b = outputs['deblur'][2][batch_idx].clip(0, 1) * 255.0
    c = seq['deblur'][frame][batch_idx]*255.0
    d = seq['image'][frame-1][batch_idx]*255.0
    e = (1 - torch.argmax(outputs['segment'][2][batch_idx], 0).repeat(3, 1, 1))*255.0
    f = flow_to_image(outputs['flow'][2][batch_idx])
    g = d2[2][batch_idx].clip(0, 1)*255.0
    h = seq['segment'][frame][batch_idx].repeat(3, 1, 1)*255.0
    i = flow_to_image(seq['flow'][frame][batch_idx].permute(2, 0, 1))
    grid = make_grid([a,b,c,d,e,f,g,h,i], nrow=3, padding=20)
    grid = torch.nn.functional.interpolate(grid.unsqueeze(0), scale_factor=.25).squeeze(0)
    import cv2
    cv2.imwrite('./here.png', cv2.cvtColor(grid.permute(1,2,0).numpy(), cv2.COLOR_BGR2RGB))

    return cv2.cvtColor(grid.permute(1,2,0).numpy(), cv2.COLOR_BGR2RGB)

def homo2offsets(homo, h, w):
    """

    Args:
        homo: Torch.Tensor [B, 3, 3]
        h: Int, height
        w: Int, width

    Returns:
        offsets: [B, 4, 2]
    """
    corners = torch.tensor([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).float()
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
    corners = torch.tensor([[[0, 0], [w, 0], [0, h], [w, h]]]).float()
    offsets = offsets.view(offsets.shape[0], 4, 2)
    offsets = corners + offsets
    homo = kornia.geometry.transform.get_perspective_transform(corners.repeat(offsets.shape[0], 1, 1), offsets)
    return homo

def warp_flow(input, flow, size):
    return input


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
