"""1D 합성곱 모델
"""
import torch
from torch import nn
from torch.nn import functional


class Conv1DModel(nn.Module):
    def __init__(self, input_channels: int, num_conv_layers: int) -> None:
        super(Conv1DModel, self).__init__()
        self.__conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=input_channels if i == 0 else 32,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
                for i in range(num_conv_layers)
            ]
        )
        self.__fc = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.__conv_layers:
            x = functional.relu(conv(x))
        x = functional.avg_pool1d(x, kernel_size=int(x.size(dim=2)))
        x = x.view(x.size()[0], -1)
        x = self.__fc(x)
        x = torch.sigmoid(x) * 1000
        return x


if __name__ == "__main__":
    batch_size = 2
    input_channels = 3
    num_conv_layers = 2

    model = Conv1DModel(input_channels, num_conv_layers)
    x = torch.rand(batch_size, input_channels, 10)

    print(x)
    print(model(x))
