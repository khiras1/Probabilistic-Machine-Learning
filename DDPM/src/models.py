import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.unets.unet_2d import UNet2DModel

from src.inception import fid_inception_v3

class LinearResBlock(nn.Module):
    def __init__(self, dim):
        super(LinearResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class SmallDenoiserNetwork(nn.Module):
    def __init__(self, hidden_dim=64, resblocks=2, global_skip_connection=False):
        super(SmallDenoiserNetwork, self).__init__()
        self.global_residual = global_skip_connection
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            *[LinearResBlock(hidden_dim) for _ in range(resblocks)],
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x_t, t):
        t = t.float().view(-1, 1) / 1000.0
        net_input = torch.cat([x_t, t], dim=1)
        if not self.global_residual:
            return self.net(net_input)
        return self.net(net_input) + x_t


class LargeConvDenoiserNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: list[int] = [64, 128, 256, 512, 1024],
        layers_per_block: int = 2,
        downblock: str = 'ResnetDownsampleBlock2D',
        upblock: str = 'ResnetUpsampleBlock2D',
        add_attention: bool = True,
        attention_head_dim: int = 64,
        low_condition: bool = False,
        timestep_condition: bool = True,
        global_skip_connection: bool = True,
        num_class_embeds: int | None = None,
    ):
        super().__init__()
        self.low_condition = low_condition
        self.timestep_condition = timestep_condition
        self.global_skip_connection = global_skip_connection
        self.divide_factor = 2 ** len(channels)

        in_channels = 2 * in_channels if self.low_condition else in_channels

        self.backbone = UNet2DModel(
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=channels,
            layers_per_block=layers_per_block,
            down_block_types=tuple(downblock for _ in range(len(channels))),
            up_block_types=tuple(upblock for _ in range(len(channels))),
            add_attention=add_attention,
            attention_head_dim=attention_head_dim,
            num_class_embeds=num_class_embeds,
        )

    def padding(self, x):
        _, _, W, H = x.shape
        desired_width = (
            (W + self.divide_factor - 1) // self.divide_factor
        ) * self.divide_factor
        desired_height = (
            (H + self.divide_factor - 1) // self.divide_factor
        ) * self.divide_factor

        # Calculate the padding needed
        padding_w = desired_width - W
        padding_h = desired_height - H

        return F.pad(x, (0, padding_h, 0, padding_w), mode="constant", value=0), W, H

    def remove_padding(self, x, W, H):
        return x[:, :, :W, :H]

    def forward(self, x_t, t):

        # add padding to fit nearest value divisible by self.divide_factor
        x_in, W, H = self.padding(x_t)

        model_output = self.backbone(
            x_in,
            timestep=t if self.timestep_condition else 0,
        ).sample

        model_output = self.remove_padding(model_output, W, H)

        if self.global_skip_connection:
            model_output[:, :3] = model_output[:, :3] + x_t

        return model_output  # pred_x_0


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3,  # Final average pooling features
    }

    def __init__(
        self,
        output_blocks=(DEFAULT_BLOCK_INDEX,),
        resize_input=True,
        normalize_input=True,
        requires_grad=False,
    ):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, "Last possible output block index is 3"

        self.blocks = nn.ModuleList()

        inception = fid_inception_v3()

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x.squeeze(-1).squeeze(-1).cpu())

            if idx == self.last_needed_block:
                break

        if len(self.output_blocks) == 1:
            return outp[0]

        return outp