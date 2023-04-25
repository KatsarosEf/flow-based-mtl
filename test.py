import os
import numpy as np
import torch
import wandb
from argparse import ArgumentParser
from utils.dataset import MTL_TestDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from losses import OpticalFlowLoss
from metrics import OpticalFlowMetrics
from utils.transforms import ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomColorChannel,\
    ColorJitter
from utils.network_utils import model_save, model_load, gridify
import torch.nn.functional as F
import sys
sys.path.append('core')
from raft import RAFT
import tqdm

def save_outputs(args, seq_name, name, output_dict):
    return []

def evaluate(args, dataloader, model, metrics_dict):

    tasks = model.module.tasks
    metrics = [k for task in tasks for k in metrics_dict[task].metrics]
    metric_cumltive = {k: [] for k in metrics}
    model.eval()

    for seq_idx, seq in enumerate(dataloader['test']):

        seq_name = seq['meta']['paths'][0][0].split('/')[-3]
        print(seq_name)
        os.makedirs(os.path.join(args.out, seq_name, 'flows'), exist_ok=True)

        results = open(os.path.join(args.out, seq_name, 'results.txt'), 'w')

        for frame in tqdm.tqdm(range(args.prev_frames, len(seq['meta']['paths']))):

            path = [x.split('/')[-3] + '_GT_' + x.split('/')[-1][:-4] +'.png' for x in seq['meta']['paths'][frame]]
            # Load the data and mount them on cuda
            if frame == args.prev_frames:
                frames = [seq['image'][i].cuda(non_blocking=True) for i in range(frame + 1)]
            else:
                frames.append(seq['image'][frame].cuda(non_blocking=True))
                frames.pop(0)

            gt_dict = {task: seq[task][frame].cuda(non_blocking=True) if type(seq[task][frame]) is torch.Tensor else
            [e.cuda(non_blocking=True) for e in seq[task][frame]] for task in tasks}

            with torch.no_grad():
                outputs = model(frames[0], frames[1])[-1]
            outputs = dict(zip(tasks, outputs))

            name = seq['meta']['paths'][frame][0].split('/')[-1]
            save_outputs(args, seq_name, name, outputs)

            task_metrics = {task: metrics_dict[task](outputs[task], gt_dict[task]) for task in tasks}
            metrics_values = {k: torch.round((10**3 * v))/(10**3) for task in tasks for k, v in task_metrics[task].items()}

            for metric in metrics:
                metric_cumltive[metric].append(metrics_values[metric])

            results.write('\nFrame {}, EPE: {:.3f} \n'.format(name, metrics_values['EPE']))

    metric_averages = {m: sum(metric_cumltive[m])/len(metric_cumltive[m]) for m in metrics}
    print("\n[TEST] {}\n".format(' '.join(['{}: {:.3f}'.format(m, metric_averages[m]) for m in metrics])))
    wandb_logs = {"Test - {}".format(m): metric_averages[m] for m in metrics}
    wandb.log(wandb_logs)


def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"


    tasks = ['flow']

    transformations = {'test': transforms.Compose([ToTensor(), Normalize()])}
    data = {'test': MTL_TestDataset(tasks, args.data_path, 'test', args.seq_len, transform=transformations['test'])}
    loader = {'test': DataLoader(data['test'], batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=False)}


    metrics_dict = {
        'flow': OpticalFlowMetrics().to('cuda')
    }
    metrics_dict = {k: v for k, v in metrics_dict.items() if k in tasks}
    model = RAFT(args, tasks).to('cuda')
    # params, fps, flops = measure_efficiency(args)
    # print(params, fps, flops)

    model = torch.nn.DataParallel(model).to('cuda')
    # Load checkpoint
    load_model_path = os.path.join(args.out, 'ckpt_108.pth')
    _ = model_load(load_model_path, model)

    wandb.init(project='mtl-normal', entity='dst-cv', mode='disabled')
    wandb.run.name = args.out.split('/')[-1]
    wandb.watch(model)

    evaluate(args, loader, model, metrics_dict)


if __name__ == '__main__':
    parser = ArgumentParser(description='Parser of Training Arguments')

    parser.add_argument('--data', dest='data_path', help='Set dataset root_path', default='/media/efklidis/4TB/dblab_ecai', type=str)
    parser.add_argument('--model', dest='model_path', help='Set model path', default='/media/efklidis/4TB/RESULTS-ECAI/raft', type=str)
    parser.add_argument('--out', dest='out', help='Set output path', default='/media/efklidis/4TB/RESULTS-ECAI/raft', type=str)

    parser.add_argument('--block', dest='block', help='Type of block "fft", "res", "inverted", "inverted_fft" ', default='res', type=str)
    parser.add_argument('--nr_blocks', dest='nr_blocks', help='Number of blocks', default=5, type=int)

    parser.add_argument("--segment", action='store_false', help="Flag for segmentation")
    parser.add_argument("--deblur", action='store_false', help="Flag for  deblurring")
    parser.add_argument("--flow", action='store_false', help="Flag for  homography estimation")
    parser.add_argument("--resume", action='store_true', help="Flag for resume training")

    parser.add_argument('--bs', help='Set size of the batch size', default=1, type=int)
    parser.add_argument('--seq_len', dest='seq_len', help='Set length of the sequence', default=None, type=int)
    parser.add_argument('--prev_frames', dest='prev_frames', help='Set number of previous frames', default=1, type=int)

    parser.add_argument('--save_every', help='Save model every n epochs', default=1, type=int)


    args = parser.parse_args()

    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(args)
