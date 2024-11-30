"""N채널 다층 퍼셉트론 모델
"""
import torch
from torch import nn
from torch.nn import functional


class MLP(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super(MLP, self).__init__()
        self.__fc1 = nn.Linear(input_channels, 128)
        self.__fc2 = nn.Linear(128, 64)
        self.__fc3 = nn.Linear(64, 32)
        self.__fc4 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = functional.relu(self.__fc1(x))
        x = functional.relu(self.__fc2(x))
        x = functional.relu(self.__fc3(x))
        x = self.__fc4(x)
        x = torch.sigmoid(x) * 1000
        return x


if __name__ == "__main__":
    batch_size = 2
    input_channels = 3

    model = MLP(input_channels)
    x = torch.rand(batch_size, input_channels)

    print(x)
    print(model(x))
