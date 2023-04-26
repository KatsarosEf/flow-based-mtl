import os
import torch
import tqdm
import wandb
from tqdm import tqdm

from argparse import ArgumentParser
from utils.dataset import MTL_TestDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from metrics import SegmentationMetrics, DeblurringMetrics, OpticalFlowMetrics
from utils.transforms import ToTensor, Normalize



def evaluate(args, dataloader, metrics_dict):
    tasks = ['segment', 'deblur', 'flow']
    metrics = [k for task in tasks for k in metrics_dict[task].metrics]
    metric_cumltive = {k: [] for k in metrics}

    outputs = {}
    for seq_idx, seq in enumerate(dataloader['test']):

        seq_name = seq['meta']['paths'][0][0].split('/')[-3]
        print(seq_name)
        for frame in tqdm(range(args.prev_frames, len(seq['meta']['paths']))):

            path = [x.split('/')[-3] + '_GT_' + x.split('/')[-1][:-4] +'.png' for x in seq['meta']['paths'][frame]]
            # Load the data and mount them on cuda
            if frame == args.prev_frames:
                frames = [seq['image'][i].cuda(non_blocking=True) for i in range(frame + 1)]
            else:
                frames.append(seq['image'][frame].cuda(non_blocking=True))
                frames.pop(0)

            gt_dict = {task: seq[task][frame].cuda(non_blocking=True) if type(seq[task][frame]) is torch.Tensor else
            [e.cuda(non_blocking=True) for e in seq[task][frame]] for task in tasks}


            outputs['flow'] = torch.zeros((1, 2, 800, 800)).cuda()
            outputs['deblur'] = frames[1].cuda()
            outputs['segment'] = torch.cat([torch.zeros((1,1,800,800)).cuda(), torch.ones((1,1,800,800)).cuda()], 1)

            task_metrics = {task: metrics_dict[task](outputs[task], gt_dict[task]) for task in tasks}
            metrics_values = {k: torch.round((10**3 * v))/(10**3) for task in tasks for k, v in task_metrics[task].items()}

            for metric in metrics:
                metric_cumltive[metric].append(metrics_values[metric])

    metric_averages = {m: sum(metric_cumltive[m])/len(metric_cumltive[m]) for m in metrics}
    print("\n[TEST] {}\n".format(' '.join(['{}: {:.3f}'.format(m, metric_averages[m]) for m in metrics])))
    wandb_logs = {"Test - {}".format(m): metric_averages[m] for m in metrics}
    wandb.log(wandb_logs)


def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"


    tasks = ['segment', 'deblur', 'flow']

    transformations = {'test': transforms.Compose([ToTensor(), Normalize()])}
    data = {'test': MTL_TestDataset(tasks, args.data_path, 'val', args.seq_len, transform=transformations['test'])}
    loader = {'test': DataLoader(data['test'], batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=False)}


    metrics_dict = {
        'segment': SegmentationMetrics().to('cuda'),
        'deblur': DeblurringMetrics().to('cuda'),
        'flow': OpticalFlowMetrics().to('cuda')}
    metrics_dict = {k: v for k, v in metrics_dict.items() if k in tasks}


    wandb.init(project='mtl-normal', entity='dst-cv', mode='disabled')
    wandb.run.name = args.out.split('/')[-1]
    evaluate(args, loader, metrics_dict)


if __name__ == '__main__':
    parser = ArgumentParser(description='Parser of Training Arguments')

    parser.add_argument('--data', dest='data_path', help='Set dataset root_path', default='/media/efklidis/4TB/dblab_ecai', type=str) #/media/efklidis/4TB/ # ../raid/data_ours_new_split
    parser.add_argument('--out', dest='out', help='Set output path', default='/media/efklidis/4TB/results/MICCAI-2/', type=str)
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
