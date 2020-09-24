"""
Based on: https://github.com/rraileanu/auto-drac
"""
import torch.nn as nn
import torch.nn.functional as F


def apply_init_(modules):
    """
    Initialize NN modules
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    """
    Flatten a tensor
    """
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """
    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size)
        additional_padding = int(total_padding % 2 != 0)
        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(input,
                            self.weight,
                            self.bias,
                            self.stride,
                            padding=0,
                            dilation=self.dilation,
                            groups=self.groups)
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(input,
                        self.weight,
                        self.bias,
                        self.stride,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=self.dilation,
                        groups=self.groups)


class ResidualBlock(nn.Module):
    """
    Residual Network Block
    """
    def __init__(self, n_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1, 1))
        self.stride = stride

        apply_init_(self.modules())

        self.train()

    def forward(self, x):
        identity = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        return out


class ImpalaCNN(nn.Module):
    """
    Impala CNN.

    For helpful image of network:
    https://medium.com/aureliantactics/custom-models-with-baselines-impala-cnn-cnns-with-features-and-contra-3-hard-mode-811dbdf2dff9

    Arguments:
    ----------
    num_inputs : `int`
        Number of channels in the input image.
    """
    def __init__(self, num_inputs, channels=[16, 32, 32]):
        super(ImpalaCNN, self).__init__()

        # define Impala CNN
        self.layer1 = self._make_layer(num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.flatten = Flatten()
        self.relu = nn.ReLU()

        # initialise all conv modules
        apply_init_(self.modules())

        # put model into train mode
        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = list()

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(ResidualBlock(out_channels))
        layers.append(ResidualBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        return x
