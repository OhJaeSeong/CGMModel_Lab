"""N채널 선형회귀 모델
"""
import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    """N채널 선형회귀 모델"""

    def __init__(
        self, input_size: int, channel_size: int, output_mode: str = None
    ) -> None:
        """생성자
        Args:
            input_size (int): 입력 크기
            channel_size (int): 채널 크기
            output_mode (str, optional): 출력 모드. 기본값 "None".

        Note:
            output_mode가 "sum"이면 N개의 선형회귀 모델의 출력을 모두 더한 값을 출력합니다.
            output_mode가 "mean"이면 N개의 선형회귀 모델의 출력의 평균을 출력합니다.
            output_mode가 "concat"이면 N개의 선형회귀 모델의 출력을 모두 연결한 값을 출력합니다.
            output_mode가 "None"이면 N개의 선형회귀 모델의 출력을 리스트로 출력합니다.
        """
        super(LinearRegressionModel, self).__init__()
        self.__output_mode = output_mode
        self.__channel_size = channel_size
        self.__linears = nn.ModuleList(
            [nn.Linear(input_size, 1) for _ in range(channel_size)]
        )

    def forward(self, x) -> torch.Tensor:
        """순전파 함수"""
        outputs = [self.__linears[i](x[:, i, :]) for i in range(self.__channel_size)]
        outputs_concat = torch.cat(outputs, dim=1)
        if self.__output_mode == "sum":
            sum_output = torch.sum(outputs_concat, dim=1, keepdim=True)
            sigmoid_output = torch.sigmoid(sum_output)
            return sigmoid_output * 1000
        elif self.__output_mode == "mean":
            mean_output = torch.mean(outputs_concat, dim=1, keepdim=True)
            sigmoid_output = torch.sigmoid(mean_output)
            return sigmoid_output * 1000
        elif self.__output_mode == "concat":
            return outputs_concat
        else:
            return outputs
        return


if __name__ == "__main__":
    batch_size = 2
    input_size = 1
    channel_size = 3

    model = LinearRegressionModel(input_size, channel_size, "sum")
    x = torch.rand(batch_size, channel_size, input_size)

    print(x)
    print(model(x))
