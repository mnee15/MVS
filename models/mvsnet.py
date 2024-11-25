import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(1, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x


class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)

        x = self.prob(x)

        return x


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(2, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        _, _, h, w = depth_init.shape
        depth_init = F.interpolate(depth_init, size=(h*4, w*4), mode="bilinear")
        concat = torch.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


class BayesianRegressionHead2D(nn.Module): 
    def __init__(self, input_dim=1):
        super(BayesianRegressionHead2D, self).__init__()

        self.conv_mu = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, 1, 3, 1, 1)
        )
        self.conv_var_sigma = nn.Sequential(      
            nn.Conv2d(input_dim, input_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, 1, 3, 1, 1),
        )


    def forward(self, x, depth_values):
        depth_values_mat = depth_values.repeat(x.shape[2], x.shape[3], 1, 1).permute(2, 3, 0, 1)
        depth_min, depth_max = depth_values[:, 0], depth_values[:, -1]
        depth_values_mat = (depth_values_mat - depth_min) / (depth_max - depth_min)
        x = x * depth_values_mat

        # mu, sigma
        mu = self.conv_mu(x) 
        var = self.conv_var_sigma(x)
        var = F.elu(var) + 1.0 + 1e-10
        sigma = torch.sqrt(var)

        return mu, sigma


class PixelwiseNet(nn.Module):
    # From PatchmatchNet
    """Pixelwise Net: A simple pixel-wise view weight network, composed of 1x1x1 convolution layers
    and sigmoid nonlinearities, takes the initial set of similarities to output a number between 0 and 1 per
    pixel as estimated pixel-wise view weight.
    1. The similarity is calculated by ref_feature and other source_features warped by differentiable_warping
    2. The learned pixel-wise view weight is estimated in the first iteration of Patchmatch and kept fixed in the
    matching cost computation.
    """

    def __init__(self, in_channels=1) -> None:
        """Initialize method
        Args:
            in_channels: the feature channels of input
        """
        super(PixelwiseNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels=in_channels, out_channels=16, kernel_size=1, stride=1, pad=0)
        self.conv1 = ConvBnReLU3D(in_channels=16, out_channels=8, kernel_size=1, stride=1, pad=0)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        """Forward method for PixelwiseNet
        Args:
            x1: pixel-wise view weight, [B, in_channels, Ndepth, H, W]
        """
        # [B, Ndepth, H, W]
        x1 = self.conv2(self.conv1(self.conv0(x1))).squeeze(1)

        output = self.output(x1)
        del x1
        # [B,H,W]
        output = torch.max(output, dim=1)[0]

        return output.unsqueeze(1)
    

class MVSNet(nn.Module):
    def __init__(self, depth_dim=256, bayesian_mode=True):
        super(MVSNet, self).__init__()
        self.depth_dim = depth_dim

        self.feature = FeatureNet()
        self.cost_regularization = CostRegNet()

        self.pixelwise_net = PixelwiseNet()
        self.bayesian_mode = bayesian_mode
        if bayesian_mode:
            self.bayesian = BayesianRegressionHead2D(input_dim=depth_dim)

    def forward(self, imgs, proj_matrices, depth_values):
        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs]
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        ### step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
        cost_reg = self.cost_regularization(volume_variance)

        # step 3. cost volume regularization

        cost_reg = cost_reg.squeeze(1)
        prob_volume = F.softmax(cost_reg, dim=1)

        if self.bayesian_mode:
            # depth_est, sigma = self.bayesian(cost_reg, depth_values=depth_values)
            depth_est, sigma = self.bayesian(prob_volume, depth_values=depth_values)
            return {"depth": depth_est, "sigma": sigma, "prob_volume": prob_volume}
        else:
            depth_est = depth_regression(prob_volume, depth_values=depth_values)
            return {"depth": depth_est, "prob_volume": prob_volume}


def mvsnet_loss(depth_est, depth_gt, mask):
    mask = mask > 0.5
    depth_est = depth_est.squeeze(1)

    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)


def gaussian_distribution(center, sigma, size):
    x = torch.arange(size, device=center.device).float()
    distribution = torch.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    return distribution / distribution.sum()


def entropy_loss(prob_volume, depth_gt, mask, depth_value):
    shape = depth_gt.shape  # B,H,W
    depth_num = depth_value.shape[1]

    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2, 3, 0, 1)  # B,N,H,W
    else:
        depth_value_mat = depth_value

    gt_index_image = torch.argmin(torch.abs(depth_value_mat - depth_gt.unsqueeze(1)), dim=1)
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1)  # B, 1, H, W

    sigma = 1

    indices = torch.arange(depth_num, device=gt_index_image.device).view(1, depth_num, 1, 1)
    gt_index_volume = torch.exp(-0.5 * ((indices - gt_index_image) / sigma) ** 2)
    gt_index_volume = gt_index_volume / (sigma * torch.sqrt(torch.tensor(2.0 * torch.pi, device=gt_index_volume.device)))
    gt_index_volume = gt_index_volume / (torch.sum(gt_index_volume, dim=1, keepdim=True))
    
    ###### Compute KL divergence
    kl_div_image = torch.sum(gt_index_volume * torch.log((gt_index_volume + 1e-6) / (prob_volume + 1e-6)), dim=1)
    masked_kl_div = torch.mean(kl_div_image[mask])

    ###### Compute CE
    # cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-6), dim=1).squeeze(1)
    # masked_kl_div = torch.mean(cross_entropy_image[mask_true])

    return masked_kl_div

def mvsnet_loss_KL(depth_est, depth_gt, mask, prob_volume, depth_value): 
    depth_est = depth_est.squeeze(1)
    mask = mask > 0.5
    l1_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True) 
    
    kl_loss = entropy_loss(prob_volume, depth_gt, mask, depth_value)

    return l1_loss, kl_loss

def NLLKLLoss(depth_est, sigma, depth_gt, mask, prob_volume, depth_value, beta=0.5):
    mask = mask > 0.5
    depth_est = depth_est.squeeze(1)
    sigma = sigma.squeeze(1)

    var = torch.square(sigma)
    var[var < 1e-10] = 1e-10

    kl_loss = entropy_loss(prob_volume, depth_gt, mask, depth_value)
    
    ## bayesian head -> depth_gt: 0~1
    depth_min, depth_max = depth_value[:, 0], depth_value[:, -1]
    depth_gt = (depth_gt - depth_min) / (depth_max - depth_min)

    nll_loss = (torch.square(depth_est - depth_gt) / (2 * var)) + (0.5 * torch.log(var))

    if beta > 0:
        nll_loss = nll_loss * (sigma.detach() ** beta)

    nll_loss = nll_loss.squeeze(1)

    return torch.mean(nll_loss[mask]), kl_loss