from typing import Any, List

import torch
import torchvision


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize: bool = True) -> None:
        """
        Initialize the VGG Perceptual Loss module.

        Args:
            resize (bool): Set to True to resize input images to 224x224. Defaults to True.
        """
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        feature_layers: List[int] = [0, 1, 2, 3],
        style_layers: List[Any] = [],
    ) -> torch.Tensor:
        """
        Compute the perceptual loss between input and target images.

        Args:
            input (torch.Tensor): The input tensor of shape (N, C, H, W) where N is the batch size,
                                  C is the number of channels, and H, W are height and width.
            target (torch.Tensor): The target tensor of the same shape as input for comparison.
            feature_layers (List[int]): Indices of VGG layers to be used for feature loss computation.
                                        Defaults to [0, 1, 2, 3].
            style_layers (List[Any]): Indices of VGG layers to be used for style loss computation.
                                      Defaults to an empty list.

        Returns:
            torch.Tensor: The computed perceptual loss as a scalar tensor.
        """

        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(
                input, mode="bilinear", size=(224, 224), align_corners=False
            )
            target = self.transform(
                target, mode="bilinear", size=(224, 224), align_corners=False
            )
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1 (torch.Tensor): The first image as a torch.Tensor.
        img2 (torch.Tensor): The second image as a torch.Tensor.
        max_val (float, optional): The maximum possible pixel value (default is 1.0).

    Returns:
        torch.Tensor: The PSNR value as a torch.Tensor.
    """
    mse = torch.nn.functional.mse_loss(img1, img2)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr
