import torch
from torch import nn
import torch.nn.functional as F


class SemanticSegmentationLoss(nn.Module):

	def __init__(self, ce_factor=1, device='cuda'):
		super(SemanticSegmentationLoss, self).__init__()
		self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum').to(device)
		self.ce_factor = ce_factor

	def forward(self, output, gt):
		if type(output) is list:
			losses = []
			for num, elem in enumerate(output[::-1]):
				losses.append(self.cross_entropy_loss(elem, nn.functional.interpolate(gt.float().unsqueeze(1), scale_factor=1.0/(2**num)).long().squeeze(1)))
			cross_entropy_loss = sum(losses)
		else:
			cross_entropy_loss = self.cross_entropy_loss(output, gt)
		return self.ce_factor * cross_entropy_loss

class EdgeLoss(nn.Module):

	def __init__(self):
		super(EdgeLoss, self).__init__()
		k = torch.Tensor([[.05, .25, .4, .25, .05]])
		self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
		if torch.cuda.is_available():
			self.kernel = self.kernel.cuda()
		self.loss = CharbonnierLoss()

	def conv_gauss(self, img):
		n_channels, _, kw, kh = self.kernel.shape
		img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
		return F.conv2d(img, self.kernel, groups=n_channels)

	def laplacian_kernel(self, current):
		filtered    = self.conv_gauss(current)    # filter
		down        = filtered[:,:,::2,::2]               # downsample
		new_filter  = torch.zeros_like(filtered)
		new_filter[:,:,::2,::2] = down*4                  # upsample
		filtered    = self.conv_gauss(new_filter) # filter
		diff = current - filtered
		return diff

	def forward(self, x, y):
		loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
		return loss

class MSEdgeLoss(nn.Module):

	def __init__(self, device='cuda'):
		super(MSEdgeLoss, self).__init__()
		self.loss_function = EdgeLoss().to(device)

	def forward(self, output, gt):
		if type(output) is list:
			losses = []
			for num, elem in enumerate(output[::-1]):
				losses.append(self.loss_function(elem, torch.nn.functional.interpolate(gt, scale_factor=1.0/(2**num))))
			loss = sum(losses)
		else:
			loss = self.loss_function(output, gt)
		return loss


class CharbonnierLoss(nn.Module):

	def __init__(self):
		super(CharbonnierLoss, self).__init__()
		self.eps = 1e-6

	def forward(self, X, Y):
		return torch.sqrt((X - Y) ** 2 + self.eps).sum()


class DeblurringLoss(nn.Module):

	def __init__(self, CL_factor=1, device='cuda', sobel=False, perceptual=False):

		super(DeblurringLoss, self).__init__()
		self.sobel = sobel
		self.perceptual = perceptual
		self.CL_factor = CL_factor
		self.CL = ContentLoss(mode='charbonnier').to(device)
		if sobel:
			self.E_factor = 1
			self.EL = MSEdgeLoss().to(device)
		if perceptual:
			self.P_factor = 0.002
			self.PL = ResNetPLoss().to(device)


	def forward(self, output, gt):
		if not self.sobel and not self.perceptual:
			cl_loss = self.CL(output, gt)
			return self.CL_factor * cl_loss
		else:
			cl_loss = self.CL(output, gt)
			el_loss = self.EL(output, gt)
			#pl_loss = self.PL(output, gt)
			return self.CL_factor * cl_loss + self.E_factor * el_loss #+ self.P_factor * pl_loss

class ContentLoss(nn.Module):

	def __init__(self, mode='charbonnier', device='cuda'):

		super(ContentLoss, self).__init__()
		if mode == 'l2':
			self.loss_function = nn.MSELoss(reduction='sum').to(device)
		elif mode == 'l1':
			self.loss_function = nn.L1Loss(reduction='sum').to(device)
		elif mode == 'charbonnier':
			self.loss_function = CharbonnierLoss().to(device)

	def forward(self, output, gt):
		if type(output) is list:
			losses = []
			for num, elem in enumerate(output[::-1]):
				losses.append(self.loss_function(elem, torch.nn.functional.interpolate(gt, scale_factor=1.0/(2**num))))
			loss = sum(losses)
		else:
			loss = self.loss_function(output, gt)
		return loss

IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]

class ResNetPLoss(nn.Module):
	def __init__(self, weight=1, weights_path='./models', arch_encoder='resnet50dilated', segmentation=True):
		super().__init__()
		self.impl = ModelBuilder.get_encoder(weights_path=weights_path,
											 arch_encoder=arch_encoder,
											 arch_decoder='ppm_deepsup',
											 fc_dim=2048,
											 segmentation=segmentation)
		self.impl.eval()
		for w in self.impl.parameters():
			w.requires_grad_(False)

		self.weight = weight

	def forward(self, pred, target):
		pred = (pred[2] - IMAGENET_MEAN.to(pred[2])) / IMAGENET_STD.to(pred[2])
		target = (target - IMAGENET_MEAN.to(target)) / IMAGENET_STD.to(target)

		pred_feats = self.impl(pred, return_feature_maps=True)
		target_feats = self.impl(target, return_feature_maps=True)

		result = torch.stack([F.mse_loss(cur_pred, cur_target, reduction='sum')
							  for cur_pred, cur_target
							  in zip(pred_feats, target_feats)]).sum() * self.weight
		return result


class FFTLoss(nn.Module):

	def __init__(self, mode='charbonnier', device='cuda'):

		super(FFTLoss, self).__init__()
		if mode == 'l2':
			self.loss_function = nn.MSELoss(reduction='sum').to(device)
		elif mode == 'l1':
			self.loss_function = nn.L1Loss(reduction='sum').to(device)
		elif mode == 'charbonnier':
			self.loss_function = CharbonnierLoss().to(device)

	def forward(self, output, gt):
		if type(output) is list:
			losses = []
			for num, elem in enumerate(output[::-1]):
				out_fft = torch.rfft(elem, signal_ndim=2, normalized=False, onesided=False)
				gt_fft = torch.rfft(torch.nn.functional.interpolate(gt, scale_factor=1.0/(2**num)), signal_ndim=2, normalized=False, onesided=False)
				losses.append(self.loss_function(out_fft, gt_fft))
			loss = sum(losses)
		else:
			loss = self.loss_function(output, gt)
		return loss

class MaceLoss(nn.Module):

	def __init__(self):
		super(MaceLoss, self).__init__()
		self.eps = 1e-6

	def forward(self, X, Y):
		return ((X - Y) ** 2 + self.eps).sum(dim=2).sqrt().sum()


class HomographyLoss(nn.Module):
	def __init__(self, device='cuda'):
		super(HomographyLoss, self).__init__()
		self.MaceLoss = MaceLoss().to(device)

	def forward(self, output, gt):
		if type(output) is list:
			losses = []
			for num, elem in enumerate(output[::-1]):
				losses.append(self.MaceLoss(elem, gt[1] / (2**num)))
			MaceLoss = sum(losses)
		else:
			MaceLoss = self.MaceLoss(output, gt)
		return MaceLoss

class EPELoss(nn.Module):
	def __init__(self):
		super(EPELoss, self).__init__()
		self.eps = 1e-6
	def forward(self, flow_pred, flow_gt):
		return torch.sum((flow_pred - flow_gt)**2 + self.eps, dim=1).sqrt()

class OpticalFlowLoss(nn.Module):
	def __init__(self, device='cuda'):
		super(OpticalFlowLoss, self).__init__()
		self.EPELoss = EPELoss().to(device)
	def forward(self, output, gt):
		if type(output) is list:
			losses = []
			for num, elem in enumerate(output[::-1]):
				losses.append(self.EPELoss(elem, gt[1]))
			EPELoss = sum(losses)
		else:
			EPELoss = self.EPELoss(output, gt)
		return EPELoss

