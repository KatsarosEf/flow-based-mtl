import torch
from torch.nn import Module

from torchmetrics.functional import psnr, ssim


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

class Metric(Module):

	def __init__(self):
		super(Metric, self).__init__()

	def forward(self, output, gt):
		with torch.no_grad():
			metric_value = self.compute_metric(output, gt)
		return metric_value

	def compute_metric(self, output, gt):
		return None


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


class EPE(Metric):

	def __init__(self):
		super(EPE, self).__init__()

	def compute_metric(self, output, gt):
		return torch.sum((output - gt)**2, 1).sqrt().mean()


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

class OpticalFlowMetrics(Module):

	def __init__(self):
		super(OpticalFlowMetrics, self).__init__()
		self.metrics = {
			'EPE': EPE()}

	def forward(self, output, gt):
		with torch.no_grad():
			metric_results = {metric: metric_function(output, gt) for num, (metric, metric_function) in enumerate(self.metrics.items())}
		return metric_results


