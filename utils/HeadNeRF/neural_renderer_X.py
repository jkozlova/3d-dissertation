import torch.nn as nn
import torch
from math import log, log2
import torch.nn.functional as F

from torchvision.ops import SqueezeExcitation
from kornia.filters import filter2d


class Blur(nn.Module):
    def __init__(self) -> None:
        """
        Initialize a Blur module with a fixed filter f.

        The filter is of shape (3,) and is defined as [1, 2, 1]. This filter is used
        to blur the input tensor in the forward pass.
        """
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer("f", f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Blurs the input x using a fixed filter defined by self.f.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            torch.Tensor: Blurred input tensor of shape (N, C, H, W)
        """
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)


class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_feature: int) -> None:
        """
        Initialize a PixelShuffleUpsample module.

        Args:
            in_feature (int): Input feature channels.
        """
        super().__init__()
        self.in_feature = in_feature
        self._make_layer()

    def _make_layer(self) -> None:
        """
        Initialize layers for PixelShuffleUpsample.

        This method creates a list of layers for PixelShuffleUpsample. It includes
        two convolutional layers, two dilated convolutional layers, a blur layer,
        an activation function, and two squeeze excitation layers.
        """
        self.layer_1 = nn.Conv2d(self.in_feature, self.in_feature * 2, 1, 1, padding=0)
        self.layer_2 = nn.Conv2d(
            self.in_feature * 2, self.in_feature * 4, 1, 1, padding=0
        )

        self.layer_dilated_1 = nn.Conv2d(
            self.in_feature, self.in_feature * 2, 3, 1, padding=2, dilation=2
        )
        self.layer_dilated_2 = nn.Conv2d(
            self.in_feature * 2, self.in_feature * 4, 3, 1, padding=2, dilation=2
        )
        self.blur_layer = Blur()
        self.actvn = nn.LeakyReLU(0.2, inplace=True)
        self.SE_1 = SqueezeExcitation(self.in_feature, self.in_feature)
        self.SE_dilated = SqueezeExcitation(self.in_feature, self.in_feature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PixelShuffleUpsample module, applying a series of
        convolutions, activations, and pixel shuffling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Upsampled and blurred tensor of shape (N, C_out, H*2, W*2).
        """
        y = self.SE_dilated(x)

        y = self.actvn(self.layer_dilated_1(y))
        y = self.actvn(self.layer_dilated_2(y))

        out = self.SE_1(x)
        out = self.actvn(self.layer_1(out))
        out = self.actvn(self.layer_2(out))

        out = out + y
        out = F.pixel_shuffle(out, 2)
        out = self.blur_layer(out)

        return out


class NeuralRenderer(nn.Module):
    def __init__(
        self,
        bg_type: str = "white",
        feat_nc: int = 256,
        out_dim: int = 3,
        final_actvn: bool = True,
        min_feat: int = 32,
        featmap_size: int = 32,
        img_size: int = 256,
        **kwargs
    ) -> None:
        """
        Initialize the NeuralRenderer module.

        Args:
            bg_type (str): Type of background for rendering. Defaults to "white".
            feat_nc (int): Number of feature channels in the neural renderer.
                Defaults to 256.
            out_dim (int): Output dimension of the neural renderer. Defaults to 3.
            final_actvn (bool): Whether to use an activation function in the final layer.
                Defaults to True.
            min_feat (int): Minimum number of features in the neural renderer. Defaults to 32.
            featmap_size (int): Size of the feature map used in the neural renderer.
                Defaults to 32.
            img_size (int): Size of the output image. Defaults to 256.
        """
        super().__init__()

        self.bg_type = bg_type
        self.featmap_size = featmap_size
        self.final_actvn = final_actvn
        self.n_feat = feat_nc
        self.out_dim = out_dim
        self.n_blocks = int(log2(img_size) - log2(featmap_size))
        self.min_feat = min_feat
        self._make_layer()
        self._build_bg_featmap()

    def _build_bg_featmap(self) -> None:
        """
        Build a feature map for the background.

        The feature map is a 4D tensor of shape (1, n_feat, featmap_size, featmap_size).
        The values of the tensor are either all 1 (for white background) or all 0 (for black background).
        The tensor is registered as a torch.nn.Parameter.
        """
        if self.bg_type == "white":
            bg_featmap = torch.ones(
                (1, self.n_feat, self.featmap_size, self.featmap_size),
                dtype=torch.float32,
            )
        elif self.bg_type == "black":
            bg_featmap = torch.zeros(
                (1, self.n_feat, self.featmap_size, self.featmap_size),
                dtype=torch.float32,
            )
        else:
            bg_featmap = None
            print("Error bg_type")
            exit(0)

        self.register_parameter("bg_featmap", torch.nn.Parameter(bg_featmap))

    def get_bg_featmap(self) -> torch.Tensor:
        """
        Returns the background feature map tensor.

        Returns:
            torch.Tensor: A tensor of shape (1, n_feat, featmap_size, featmap_size)
        """
        return self.bg_featmap

    def _make_layer(self) -> None:
        """
        Initializes the layers required for the NeuralRenderer.

        This method creates several lists of layers, including upsampling layers
        for feature maps, layers for converting features to RGB, and convolutional
        layers for feature transformation. The layers are structured to perform
        upsampling, blurring, and activation functions, preparing the model for
        the forward pass.

        Attributes:
            feat_upsample_list (nn.ModuleList): A list of PixelShuffleUpsample modules
                for upsampling feature maps.
            rgb_upsample (nn.Sequential): A sequential model consisting of an upsampling
                layer and a blur layer for RGB images.
            feat_2_rgb_list (nn.ModuleList): A list of convolutional layers for converting
                feature maps to RGB images.
            feat_layers (nn.ModuleList): A list of convolutional layers for feature
                transformation across different scales.
            actvn (nn.LeakyReLU): Leaky ReLU activation function with a negative slope of 0.2.
        """
        self.feat_upsample_list = nn.ModuleList(
            [
                PixelShuffleUpsample(max(self.n_feat // (2 ** (i)), self.min_feat))
                for i in range(self.n_blocks)
            ]
        )

        self.rgb_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), Blur()
        )

        self.feat_2_rgb_list = nn.ModuleList(
            [nn.Conv2d(self.n_feat, self.out_dim, 1, 1, padding=0)]
            + [
                nn.Conv2d(
                    max(self.n_feat // (2 ** (i + 1)), self.min_feat),
                    self.out_dim,
                    1,
                    1,
                    padding=0,
                )
                for i in range(0, self.n_blocks)
            ]
        )

        self.feat_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    max(self.n_feat // (2 ** (i)), self.min_feat),
                    max(self.n_feat // (2 ** (i + 1)), self.min_feat),
                    1,
                    1,
                    padding=0,
                )
                for i in range(0, self.n_blocks)
            ]
        )

        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NeuralRenderer, transforming the input feature map into
        an RGB image. The process includes upsampling, applying convolutional layers,
        and optional activation functions.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W), where N is the batch size,
                            C is the number of channels, H and W are height and width.

        Returns:
            torch.Tensor: The output RGB image tensor of shape (N, out_dim, img_size, img_size).
        """
        rgb = self.rgb_upsample(self.feat_2_rgb_list[0](x))
        net = x
        for idx in range(self.n_blocks):
            hid = self.feat_layers[idx](self.feat_upsample_list[idx](net))
            net = self.actvn(hid)

            rgb = rgb + self.feat_2_rgb_list[idx + 1](net)
            if idx < self.n_blocks - 1:
                rgb = self.rgb_upsample(rgb)

        if self.final_actvn:
            rgb = torch.sigmoid(rgb)

        return rgb
