### This loss function modified based on the Original ZoeDepth Loss Class

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np

KEY_OUTPUT = 'metric_depth'

def extract_key(prediction, key):
    if isinstance(prediction, dict):
        return prediction[key]
    return prediction

"""This implementation changed several original loss classes, try different values of beta in SILogLoss
and rewrote the Grad1Loss to optimize the code efficiency. Finally, OrdinalRegressionLoss class is
then modified to Removed the object base class since it is not needed in Python 3. Changed the variable
names to conform to PEP8 naming conventions."""


class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""

    def __init__(self, beta=0.25):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            input = input[mask]
            target = target[mask]

        with amp.autocast(enabled=False):  # amp causes NaNs in this loss function
            alpha = 1e-7
            g = torch.log(input + alpha) - torch.log(target + alpha)

            # n, c, h, w = g.shape
            # norm = 1/(h*w)
            # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

            Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)

            loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print("Nan SILog loss")
            print("input:", input.shape)
            print("target:", target.shape)
            print("G", torch.sum(torch.isnan(g)))
            print("Input min max", torch.min(input), torch.max(input))
            print("Target min max", torch.min(target), torch.max(target))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))

        if not return_interpolated:
            return loss

        return loss, intr_input


def grad(x):
    # Compute the differences in the x and y directions
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]

    # Compute the magnitude of the gradient using the Pythagorean theorem
    mag = torch.sqrt(diff_x ** 2 + diff_y ** 2)

    # Compute the angle of the gradient using the arctangent function
    angle = torch.atan2(diff_y, diff_x + 1e-10)

    return mag, angle


def grad_mask(mask):
    return mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]

class GradL1Loss(nn.Module):
    """Gradient loss"""

    def __init__(self):
        super().__init__()

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        # Extract the output from the input dict
        input = input[KEY_OUTPUT]

        # If input and target have different sizes, interpolate the input
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)

        # Compute the gradients of the target and input images
        grad_gt = grad(target)
        grad_pred = grad(input)

        # Create a binary mask for the gradient tensors
        mask_g = grad_mask(mask)

        # Compute the L1 loss between the masked gradients of the predicted and ground truth images
        loss = nn.functional.l1_loss(grad_pred[0][mask_g], grad_gt[0][mask_g]) \
             + nn.functional.l1_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])

        # Return the loss or a tuple containing the loss and the interpolated input tensor
        if not return_interpolated:
            return loss
        else:
            return loss, input


class OrdinalRegressionLoss:

    def __init__(self, ord_num, beta, discretization="SID"):
        self.ord_num = ord_num
        self.beta = beta
        self.discretization = discretization

    def _create_ord_label(self, gt):
        N, _, H, W = gt.shape

        ord_c0 = torch.ones(N, self.ord_num, H, W).to(gt.device)

        if self.discretization == "SID":
            label = self.ord_num * torch.log(gt) / np.log(self.beta)
        else:
            label = self.ord_num * (gt - 1.0) / (self.beta - 1.0)

        label = label.long()

        mask = torch.linspace(
            0, self.ord_num - 1, self.ord_num, requires_grad=False
        ).view(1, self.ord_num, 1, 1).to(gt.device)

        mask = mask.repeat(N, 1, H, W).contiguous().long()
        mask = (mask > label)

        ord_c0[mask] = 0
        ord_c1 = 1 - ord_c0

        ord_label = torch.cat((ord_c0, ord_c1), dim=1)
        return ord_label, mask

    def __call__(self, prob, gt):
        valid_mask = gt > 0.
        ord_label, mask = self._create_ord_label(gt)

        entropy = -prob * ord_label
        loss = torch.sum(entropy, dim=1)[valid_mask.squeeze(1)]

        return loss.mean()


class DiscreteNLLLoss(nn.Module):
    """Cross entropy loss"""

    def __init__(self, min_depth=1e-3, max_depth=10, depth_bins=64):
        super(DiscreteNLLLoss, self).__init__()
        self.name = 'CrossEntropy'
        self.ignore_index = -(depth_bins + 1)
        # self._loss_func = nn.NLLLoss(ignore_index=self.ignore_index)
        self._loss_func = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_bins = depth_bins
        self.alpha = 1
        self.zeta = 1 - min_depth
        self.beta = max_depth + self.zeta

    def quantize_depth(self, depth):
        # depth : N1HW
        # output : NCHW

        # Quantize depth log-uniformly on [1, self.beta] into self.depth_bins bins
        depth = torch.log(depth / self.alpha) / np.log(self.beta / self.alpha)
        depth = depth * (self.depth_bins - 1)
        depth = torch.round(depth)
        depth = depth.long()
        return depth

    def _dequantize_depth(self, depth):
        """
        Inverse of quantization
        depth : NCHW -> N1HW
        """
        # Get the center of the bin

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        # assert torch.all(input <= 0), "Input should be negative"

        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        # assert torch.all(input)<=1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        target = self.quantize_depth(target)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            # Set the mask to ignore_index
            mask = mask.long()
            input = input * mask + (1 - mask) * self.ignore_index
            target = target * mask + (1 - mask) * self.ignore_index

        input = input.flatten(2)  # N, nbins, H*W
        target = target.flatten(1)  # N, H*W
        loss = self._loss_func(input, target)

        if not return_interpolated:
            return loss
        return loss, intr_input


def compute_scale_and_shift(prediction, target, mask):
    """
    Computes the scale and shift required to match the prediction to the target,
    where both are masked by the provided mask.

    Args:
        prediction (torch.Tensor): Predicted values.
        target (torch.Tensor): Target values.
        mask (torch.Tensor): Mask to apply to the prediction and target tensors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The scale and shift factors.
    """
    a_00 = torch.sum(mask * prediction * prediction, dim=(1, 2))
    a_01 = torch.sum(mask * prediction, dim=(1, 2))
    a_11 = torch.sum(mask, dim=(1, 2))

    b_0 = torch.sum(mask * prediction * target, dim=(1, 2))
    b_1 = torch.sum(mask * target, dim=(1, 2))

    det = a_00 * a_11 - a_01 * a_01
    valid = det > 0

    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1



class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SSILoss"

    def forward(self, prediction, target, mask, interpolate=True, return_interpolated=False):

        if prediction.shape[-1] != target.shape[-1] and interpolate:
            prediction = nn.functional.interpolate(prediction, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = prediction
        else:
            intr_input = prediction

        prediction, target, mask = prediction.squeeze(), target.squeeze(), mask.squeeze()
        assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

        scale, shift = compute_scale_and_shift(prediction, target, mask)

        scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        loss = nn.functional.l1_loss(scaled_prediction[mask], target[mask])
        if not return_interpolated:
            return loss
        return loss, intr_input


if __name__ == '__main__':
    # Tests for DiscreteNLLLoss
    celoss = DiscreteNLLLoss()
    print(celoss(torch.rand(4, 64, 26, 32) * 10, torch.rand(4, 1, 26, 32) * 10, ))

    d = torch.Tensor([6.59, 3.8, 10.0])
    print(celoss.dequantize_depth(celoss.quantize_depth(d)))

