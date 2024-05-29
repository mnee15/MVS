import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import cv2
from torchmetrics.functional.image import image_gradients


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    #cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=0, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
    
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss


def gaussian(window_size, sigma):
    """
    Generates a list of Tensor values drawn from a gaussian distribution with standard
    diviation = sigma and sum of all elements = 1.

    Length of list = window_size
    """    
    gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):

    # Generate an 1D tensor containing values sampled from a gaussian distribution
    # _1d_window : (window_size, 1)
    # sum of _1d_window = 1
    _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
    
    # Converting to 2D  : _1d_window (window_size, 1) @ _1d_window.T (1, window_size)
    # _2d_window : (window_size, window_size)
    # sum of _2d_window = 1
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
     
    # expand _2d_window to window size
    # window : (channel, 1, window_size, window_size)
    window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

    return window

def SSIM(img1, img2, window_size=11, val_range=255, window=None, size_average=True, full=False):

    # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),    
    L = val_range
    
    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    # if window is not provided, init one
    if window is None: 
        # window should be at least 11x11 
        real_size = min(window_size, height, width) 
        window = create_window(real_size, channel=channels).to(img1.device)
    
    # calculating the mu parameter (locally) for both images using a gaussian filter 
    # calculates the luminosity params
    pad = window_size//2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 =  F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability 
    C1 = (0.01 ) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03 ) ** 2 

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1  
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1 
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean() 
    else: 
        ret = ssim_score.mean(1).mean(1).mean(1)
    
    if full:
        return ret, contrast_metric
    
    return ret


class ShrinkageLoss(nn.Module):
    def __init__(self, alpha=10, c=0.2, reduction='mean'):
        # alpha: shrinkage speed / c: localization
        super(ShrinkageLoss, self).__init__()
        self.alpha = alpha
        self.c = c
        self.reduction = reduction
    
    def forward(self, output, target):
        MSELoss = nn.MSELoss()
        mse = MSELoss(output, target)

        # return math.pow(mse.item(), (2 + self.gamma) / 2)
        return mse / (1 + math.exp(self.alpha * (self.c - math.sqrt(mse))))


class GradLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, input):
        dy1, dx1 = image_gradients(output)
        dy2, dx2 = image_gradients(input)

        # loss = nn.L1Loss()
        loss = nn.MSELoss()

        return (loss(dx1, dx2) + loss(dy1, dy2)) / 2
        # return loss(dx1, dx2)


class GeoConstLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, alpha=1, beta=1, thrs=0.5):
        
        # const_x_map = input[:, :, 1:, :] - input[:, :, :-1, :]
        # const_y_map = input[:, :, :,1:] - input[:, :, :, :-1]
        # const_x_map[const_x_map < thrs] = 0
        # const_y_map[const_y_map < thrs] = 0
        geo_const_x = (input[:, :, 1:, :] - input[:, :, :-1, :]).mean()
        geo_const_y = (input[:, :, :,1:] - input[:, :, :, :-1]).mean()

        # return alpha * geo_const_x + beta * geo_const_y
        return geo_const_x + geo_const_y


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        self.filter = f.expand(1, 1, 3, 3)

    def forward(self, output, target):
        edge_pred = F.conv2d(output, self.filter, stride=1, padding=1)
        edge_gt = F.conv2d(target, self.filter, stride=1, padding=1)

        l2_loss = nn.MSELoss()

        return l2_loss(edge_pred, edge_gt)



_criterion_entrypoints = {
    "MSE": nn.MSELoss,
    "Huber": nn.HuberLoss,
    "Focal": FocalLoss,
    "CE": nn.CrossEntropyLoss, 
    "CE_Dice": CE_DiceLoss,
    "Shrinkage": ShrinkageLoss,
    "GradLoss": GradLoss,
    "GeoLoss": GeoConstLoss
}


def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn()
    else:
        raise RuntimeError("Unknown loss (%s)" % criterion_name)
    return criterion