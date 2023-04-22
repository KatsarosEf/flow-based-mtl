import torch
from torch import nn
import torch.nn.functional as F


class SemanticSegmentationLoss(nn.Module):

	def __init__(self, args, ce_factor=1, device='cuda'):
		super(SemanticSegmentationLoss, self).__init__()
		self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean').to(device)
		self.ce_factor = ce_factor
		self.gamma = args.gamma

	def forward(self, output, gt):
		if type(output) is list:
			losses = []
			for num, elem in enumerate(output[::-1]):
				losses.append(self.cross_entropy_loss(elem, nn.functional.interpolate(gt.float().unsqueeze(1), scale_factor=1.0/(2**num)).long().squeeze(1)))
			cross_entropy_loss = sum([losses[i]*(self.gamma**(len(output) - i - 1)) for i in range(len(output))])
		else:
			cross_entropy_loss = self.cross_entropy_loss(output, gt)
		return self.ce_factor * cross_entropy_loss

class EdgeLoss(nn.Module):

	def __init__(self):
		super(EdgeLoss, args, self).__init__()
		k = torch.Tensor([[.05, .25, .4, .25, .05]])
		self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
		if torch.cuda.is_available():
			self.kernel = self.kernel.to(args.device)
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

	def __init__(self, args, device='cuda'):
		super(MSEdgeLoss, self).__init__()
		self.loss_function = EdgeLoss(args).to(device)

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
		return torch.sqrt((X - Y) ** 2 + self.eps).mean()


class DeblurringLoss(nn.Module):

	def __init__(self, args, CL_factor=1, device='cuda', sobel=False):

		super(DeblurringLoss, self).__init__()
		self.sobel = sobel
		self.gamma = args.gamma
		self.CL_factor = CL_factor
		self.CL = ContentLoss(gamma=self.gamma, mode='charbonnier').to(device)
		if sobel:
			self.E_factor = 1
			self.EL = MSEdgeLoss().to(device)

	def forward(self, output, gt):
		if not self.sobel:
			cl_loss = self.CL(output, gt)
			return self.CL_factor * cl_loss
		else:
			cl_loss = self.CL(output, gt)
			el_loss = self.EL(output, gt)
			return self.CL_factor * cl_loss + self.E_factor * el_loss

class ContentLoss(nn.Module):

	def __init__(self, gamma, mode='charbonnier', device='cuda'):

		super(ContentLoss, self).__init__()
		self.gamma = gamma
		if mode == 'l2':
			self.loss_function = nn.MSELoss(reduction='mean').to(device)
		elif mode == 'l1':
			self.loss_function = nn.L1Loss(reduction='mean').to(device)
		elif mode == 'charbonnier':
			self.loss_function = CharbonnierLoss().to(device)

	def forward(self, output, gt):
		if type(output) is list:
			losses = []
			for num, elem in enumerate(output[::-1]):
				losses.append(self.loss_function(elem, torch.nn.functional.interpolate(gt, scale_factor=1.0/(2**num))))
			loss = sum([losses[i]*(self.gamma**(len(output) - i - 1)) for i in range(len(output))])
		else:
			loss = self.loss_function(output, gt)
		return loss

class EPELoss(nn.Module):

	def __init__(self, args):
		super(EPELoss, self).__init__()
		self.eps = 1e-6
		self.max_flow = args.max_flow

	def forward(self, flow_pred, flow_gt):
		return torch.mean((flow_pred[flow_gt.abs()<self.max_flow] - flow_gt[flow_gt.abs()<self.max_flow]).abs() + self.eps)




class OpticalFlowLoss(nn.Module):

	def __init__(self, args, device='cuda'):
		super(OpticalFlowLoss, self).__init__()
		self.args = args
		self.EPELoss = EPELoss(self.args).to(device)
		self.gamma = args.gamma

	def forward(self, output, gt):
		if type(output) is list:
			losses = []
			for num, elem in enumerate(output[::-1]):
				losses.append(self.EPELoss(elem, gt))
			EPELoss = sum([losses[i]*(self.gamma**(len(output) - i - 1)) for i in range(len(output))])
		else:
			EPELoss = self.EPELoss(output, gt)
		return EPELoss

