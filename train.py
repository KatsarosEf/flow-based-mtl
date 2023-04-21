import os
import numpy as np
import torch
import wandb
from argparse import ArgumentParser
from utils.dataset import MTL_Dataset
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from losses import DeblurringLoss, SemanticSegmentationLoss, OpticalFlowLoss
from metrics import SegmentationMetrics, DeblurringMetrics, OpticalFlowMetrics
from models.MIMOUNet.MIMOUNet import VideoMIMOUNet
from utils.transforms import ToTensor, Normalize, ColorJitter
from utils.network_utils import model_save, model_load, gridify
import torch.nn.functional as F

task_weights = {'segment': 0.1,
                'deblur': 1,
                'flow': 0.1}
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,0"


def train(args, dataloader, model, optimizer, scheduler, losses_dict, metrics_dict, epoch):
    tasks = model.module.tasks
    metrics = [k for task in tasks for k in metrics_dict[task].metrics]
    metric_cumltive = {k: [] for k in metrics}
    losses_cumltive = {k: [] for k in tasks}
    model.train()

    for seq_num, seq in enumerate(dataloader):
        for frame in range(args.prev_frames, args.seq_len):

            # Load the data and mount them on cuda
            if frame == args.prev_frames:
                frames = [seq['image'][i].to(args.device) for i in range(frame + 1)]
                m = torch.cat([seq['segment'][0].unsqueeze(1), 1 - seq['segment'][0].unsqueeze(1)], 1).float()
                m2 = [F.interpolate(m, scale_factor=0.25),
                      F.interpolate(m, scale_factor=0.5),
                      m]

                d2 = [F.interpolate(seq['image'][0], scale_factor=0.25),
                      F.interpolate(seq['image'][0], scale_factor=0.5),
                      seq['image'][0]]
            else:
                frames.append(seq['image'][frame].to(args.device))
                del frames[0]

            gt_dict = {task: seq[task][frame].to(args.device) if type(seq[task][frame]) is torch.Tensor else
            [e.to(args.device) for e in seq[task][frame]] for task in tasks}

            # import cv2
            # cv2.imwrite('./frame-t-1.jpg', frames[0][0].permute(1, 2, 0).numpy() * 255.0)
            # cv2.imwrite('./frame-t.jpg', frames[1][0].permute(1, 2, 0).numpy() * 255.0)
            # cv2.imwrite('./mask-t.jpg', gt_dict['segment'][0].numpy() * 255.0)
            # Compute model predictions, errors and gradients and perform the update
            optimizer.zero_grad()
            outputs = model(frames[0], frames[1], m2, d2)


            outputs = dict(zip(tasks, outputs))
            m2 = [x.detach() for x in outputs['segment']]
            d2 = [x.detach() for x in outputs['deblur']]

            losses = {task: losses_dict[task](outputs[task], gt_dict[task]) for task in tasks}
            loss = sum(losses.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            print('[TRAIN] [EPOCH:{}/{} ] [SEQ: {}/{}] Total Loss: {:.4f}\t{}'.format(epoch, args.epochs, seq_num+1, len(dataloader), loss, '\t'.join(
                ['{} loss: {:.4f}'.format(task, losses[task]) for task in tasks])))

            # Compute metrics for the tasks at hand
            task_metrics = {task: metrics_dict[task](outputs[task], gt_dict[task]) for task in tasks}
            metrics_values = {k: v.item() for task in tasks for k, v in task_metrics[task].items()}

            print("[TRAIN] [EPOCH:{}/{} ] {}".format(epoch, args.epochs, '\t'.join(
                ['{}: {:.4f}'.format(k, metrics_values[k]) for k in metrics])))
            for metric in metrics:
                metric_cumltive[metric].append(metrics_values[metric])
            for t in tasks:
                losses_cumltive[t].append(float(losses[t]))

    scheduler.step()
    wandb_logs = {"Train - {}".format(m): sum(metric_cumltive[m])/len(metric_cumltive[m]) for m in metrics}
    for t in tasks:
        wandb_logs['Loss: {}'.format(t)] = sum(losses_cumltive[t])/len(losses_cumltive[t])
    wandb_logs['epoch'] = epoch
    wandb_logs['lr'] = optimizer.param_groups[0]['lr']
    wandb.log(wandb_logs)



def val(args, dataloader, model, metrics_dict, epoch):

    tasks = model.module.tasks
    metrics = [k for task in tasks for k in metrics_dict[task].metrics]
    metric_cumltive = {k: [] for k in metrics}
    model.eval()
    videos2make = [[] for i in range(args.bs*args.to_visualize)]
    i = 0
    with torch.no_grad():

        for seq_idx, seq in enumerate(dataloader):
            for frame in range(args.prev_frames, args.seq_len):
                # Load the data and mount them on cuda
                if frame == args.prev_frames:
                    frames = [seq['image'][i].to(args.device) for i in range(frame + 1)]
                    m2 = [torch.zeros((frames[0].shape[0], 2, 200, 200)).to(args.device),
                          torch.zeros((frames[0].shape[0], 2, 400, 400)).to(args.device),
                          torch.zeros((frames[0].shape[0], 2, 800, 800)).to(args.device)]
                    d2 = [F.interpolate(seq['image'][0], scale_factor=0.25).to(args.device),
                          F.interpolate(seq['image'][0], scale_factor=0.5).to(args.device),
                          seq['image'][0].to(args.device)]
                else:
                    frames.append(seq['image'][frame].to(args.device))
                    del frames[0]

                gt_dict = {task: seq[task][frame].to(args.device) if type(seq[task][frame]) is torch.Tensor else
                [e.to(args.device) for e in seq[task][frame]] for task in tasks}

                outputs = model(frames[0], frames[1], m2, d2)
                outputs = dict(zip(tasks, outputs))

                task_metrics = {task: metrics_dict[task](outputs[task], gt_dict[task]) for task in tasks}
                metrics_values = {k: v.item() for task in tasks for k, v in task_metrics[task].items()}

                for metric in metrics:
                    metric_cumltive[metric].append(metrics_values[metric])

                # visualize batch size (manually coded for 2)
                if seq_idx < args.to_visualize:
                    grids = [gridify(args, seq, outputs, d2, frame, batch_idx) for batch_idx in range(args.bs)]
                    videos2make[i].append(grids[0])
                    videos2make[i+1].append(grids[1])

                m2 = outputs['segment']
                d2 = outputs['deblur']

            # Add the end of the small, 5-frame, sequences, log the videos, 2xvideos per batch
            if seq_idx < args.to_visualize:
                [wandb.log({"video_{}".format(idx): wandb.Video(np.stack((videos2make[idx])).transpose((0,3,1,2)).astype(np.uint8), fps=1)}) for idx in range(0+i, i+2)]
                i += 2

        metric_averages = {m: sum(metric_cumltive[m])/len(metric_cumltive[m]) for m in metrics}
        print("\n[VALIDATION] [EPOCH:{}/{}] {}\n".format(epoch, args.epochs,
                                                         ' '.join(['{}: {:.3f}'.format(m, metric_averages[m]) for m in metrics])))

        wandb_logs = {"Val - {}".format(m): metric_averages[m] for m in metrics}
        wandb_logs['epoch'] = epoch
        wandb.log(wandb_logs)


def main(args):


    tasks = [task for task in ['segment', 'deblur', 'flow'] if getattr(args, task)]

    transformations = {'train': transforms.Compose([ColorJitter(), ToTensor(), Normalize()]),
                       'val': transforms.Compose([ToTensor(), Normalize()])}

    data = {split: MTL_Dataset(tasks, args.data_path, split, args.seq_len, transform=transformations[split])
            for split in ['train', 'val']}

    loader = {split: DataLoader(data[split], batch_size=args.bs, shuffle=split=="train", num_workers=1, pin_memory=True, drop_last=True)
              for split in ['train', 'val']}


    losses_dict = {
        'segment': SemanticSegmentationLoss(args).to(args.device),
        'deblur': DeblurringLoss(args).to(args.device),
        'flow': OpticalFlowLoss(args).to(args.device)
    }
    losses_dict = {k: v for k, v in losses_dict.items() if k in tasks}

    metrics_dict = {
        'segment': SegmentationMetrics().to(args.device),
        'deblur': DeblurringMetrics().to(args.device),
        'flow': OpticalFlowMetrics().to(args.device)

    }
    metrics_dict = {k: v for k, v in metrics_dict.items() if k in tasks}


    model = VideoMIMOUNet(args, tasks, nr_blocks=args.nr_blocks, block=args.block).to(args.device)
    # params, fps, flops = measure_efficiency(args)
    # print(params, fps, flops)

    model = torch.nn.DataParallel(model).to(args.device)
    optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    if args.resume:
        checkpoint_file_name = 'ckpt_{}.pth'.format(args.resume_epoch) if args.resume_epoch else 'ckpt.pth'
        resume_path = os.path.join(args.out, 'models', checkpoint_file_name)
        resume_epoch = model_load(resume_path, model, optimizer, scheduler)
        start_epoch = resume_epoch + 1
    else:
        start_epoch = 1
        if 'debug' in args.out:
            os.makedirs(os.path.join(args.out, 'models'), exist_ok=True)
        else:
            os.makedirs(os.path.join(args.out, 'models'), exist_ok=True)

    wandb.init(project='mtl-normal', entity='dst-cv', mode='disabled')
    wandb.run.name = args.out.split('/')[-1]
    wandb.watch(model)



    for epoch in range(start_epoch, args.epochs+1):

        train(args, loader['train'], model, optimizer, scheduler, losses_dict, metrics_dict, epoch)

        val(args, loader['val'], model, metrics_dict, epoch)

        model_save(model, optimizer, scheduler, epoch, args)


if __name__ == '__main__':
    parser = ArgumentParser(description='Parser of Training Arguments')

    parser.add_argument('--data', dest='data_path', help='Set dataset root_path', default='/media/efklidis/4TB/dblab_ecai', type=str) # # ../raid/data_ours_new_split
    parser.add_argument('--out', dest='out', help='Set output path', default='/media/efklidis/4TB/debug-ecai-mtl', type=str)

    parser.add_argument('--block', dest='block', help='Type of block "fft", "res", "inverted", "inverted_fft" ', default='res', type=str)
    parser.add_argument('--nr_blocks', dest='nr_blocks', help='Number of blocks', default=5, type=int)

    parser.add_argument("--segment", action='store_false', help="Flag for segmentation")
    parser.add_argument("--deblur", action='store_false', help="Flag for  deblurring")
    parser.add_argument("--flow", action='store_false', help="Flag for  homography estimation")

    parser.add_argument('--lr', help='Set learning rate', default=1e-3, type=float)
    parser.add_argument('--wdecay', type=float, default=.0005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=0.99)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--bs', help='Set size of the batch size', default=4, type=int)
    parser.add_argument('--seq_len', dest='seq_len', help='Set length of the sequence', default=5, type=int)
    parser.add_argument('--max_flow', dest='max_flow', help='Set magnitude of flows to exclude from loss', default=50, type=int)
    parser.add_argument('--prev_frames', dest='prev_frames', help='Set number of previous frames', default=1, type=int)
    parser.add_argument("--device", dest='device', default="cuda", type=str)

    parser.add_argument('--epochs', dest='epochs', help='Set number of epochs', default=80, type=int)
    parser.add_argument('--save_every', help='Save model every n epochs', default=1, type=int)
    parser.add_argument("--resume", action='store_true', help="Flag for resume training")
    parser.add_argument('--resume_epoch', dest='resume_epoch', help='Number of epoch to resume', default=0, type=int)
    parser.add_argument('--to_visualize', dest='to_visualize', help='Number of mini seqs to visualize in validation', default=2, type=int)


    args = parser.parse_args()

    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(args)
