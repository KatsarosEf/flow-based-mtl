import torch
from torch.nn import Module

from torchmetrics.functional import psnr, ssim#, dice_score


def binary_segmentation_postprocessing(output: torch.Tensor):

	N, C, H, W = output.shape
	processed = torch.zeros((N, 1, H, W), dtype=torch.long).to(output.device)

	processed[output[:, 1:2] - output[:, 0:1] > 0] = 1
	return processed


def iou_score_binary(outputs: torch.Tensor, labels: torch.Tensor):
	e = 1e-6
	intersection = (outputs & labels).float().sum((-1, -2)) + e
	union = (outputs | labels).float().sum((-1, -2)) + e

	iou = intersection / union
	return iou.mean()


def dice_score_binary(outputs: torch.Tensor, labels: torch.Tensor):
	e = 1e-6
	intersection = (outputs & labels).float().sum((-1, -2)) + e
	union = (outputs | labels).float().sum((-1, -2)) + e

	dice = 2 * intersection / (intersection + union)
	return dice.mean()


def PSNR_masked(img1, img2, mask):
	mse = (img1 - img2) ** 2
	mse = (mse * mask).mean(1).sum() / mask.sum()
	return 20 * torch.log10(1.0 / torch.sqrt(mse))


def epe(flow_pred, flow_gt):
	return torch.sum((flow_pred - flow_gt)**2 + 1e-6, dim=1).sqrt() #TODO metrics needs to be taken per average, loss is to be squared only


class Metric(Module):

	def __init__(self):
		super(Metric, self).__init__()

	def forward(self, output, gt):
		with torch.no_grad():
			metric_value = self.compute_metric(output, gt)
		return metric_value

	def compute_metric(self, output, gt):
		return None

class EPE(Metric):

	def compute_metric(self, output, gt):
		return epe(output, gt)

class IoU(Metric):

	def compute_metric(self, output, gt):
		return iou_score_binary(output, gt)


class Dice(Metric):

	def compute_metric(self, output, gt):
		return dice_score_binary(output, gt)


class PSNR(Metric):

	def compute_metric(self, output, gt):
		return psnr(output.clip(0,1), gt, data_range=1.0)


class SSIM(Metric):

	def compute_metric(self, output, gt):
		return ssim(output.clip(0,1), gt, data_range=1.0)


class HomographyDistance(Metric):

	def compute_metric(self, output, gt):
		return torch.abs(output-gt).sum()


class MACE(Metric):

	def __init__(self, num):
		super(MACE, self).__init__()
		self.num = num

	def compute_metric(self, output, gt):
		return ((output[self.num] * (2 ** (2-self.num)) - gt) ** 2).sum(dim=2).sqrt().mean()

class EPE(Metric):

	def __init__(self, num):
		super(EPE, self).__init__()
		self.num = num

	def compute_metric(self, output, gt):
		return ((output[self.num] * (2 ** (2-self.num)) - gt) ** 2).sum(dim=2).sqrt().mean()


class DeblurringMetrics(Module):

	def __init__(self):
		super(DeblurringMetrics, self).__init__()
		self.metrics = {
			'psnr': PSNR(),
			'ssim': SSIM()
		}

	def forward(self, output, gt):
		with torch.no_grad():
			metric_results = {metric: metric_function(output[-1], gt) for metric, metric_function in self.metrics.items()}
		return metric_results


class SegmentationMetrics(Module):

	def __init__(self):
		super(SegmentationMetrics, self).__init__()
		self.metrics = {
			'iou': IoU(),
			'dice': Dice()
		}

	def forward(self, output, gt):
		with torch.no_grad():
			output = torch.argmax(output[-1], 1)
			metric_results = {metric: metric_function(output, gt) for metric, metric_function in self.metrics.items()}
		return metric_results


class HomographyMetrics(Module):

	def __init__(self):
		super(HomographyMetrics, self).__init__()
		self.metrics = {
			'MACE': MACE(num=2),
			'MACE_med': MACE(num=1),
			'MACE_low': MACE(num=0),
		}

	def forward(self, output, gt):
		with torch.no_grad():
			metric_results = {metric: metric_function(output, gt[1]) for metric, metric_function in self.metrics.items()}
		return metric_results

class OpticalFlowMetrics(Module):

	def __init__(self):
		super(OpticalFlowMetrics, self).__init__()
		self.metrics = {
			'EPE': EPE(num=2),
			'EPE_med': EPE(num=1),
			'EPE_low': EPE(num=0),
		}

	def forward(self, output, gt):
		with torch.no_grad():
			metric_results = {metric: metric_function(output, gt[1]) for metric, metric_function in self.metrics.items()}
		return metric_results





if __name__ == '__main__':
	prediction = torch.tensor([[[1,1,1],[1,1,1],[1,1,1]], [[0,0,0],[1,1,0],[0,0,0]]])
	gt = torch.tensor([[[1,1,1],[1,1,1],[1,1,1]], [[0,1,0],[0,1,0],[0,0,0]]])

	segmentation_metrics = SegmentationMetrics()
	print(segmentation_metrics(prediction, gt))

