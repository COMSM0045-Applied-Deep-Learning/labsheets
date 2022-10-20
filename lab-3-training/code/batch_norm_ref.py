import torch
import torch.nn as nn


class MyBatchNorm2d(nn.Module):

    def __init__(self,
                 num_features : int,
                 eps=1e-5,
                 momentum=0.1):
        """
        Args:
            num_features: input feature dimension

        For details about `eps` and `momentum`,
        see https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html

        Note:
        - Here we have not yet implemented the load/saving method,
            in which we need to take care of the running statistic.
        """
        super().__init__()
        self.gamma = nn.Parameter(
            torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(
            torch.zeros(1, num_features, 1, 1))
        self.eps = eps
        self.momentum = momentum

        # At training time, we accumulate the statistic of the dataset
        # using moving average.
        # Note these numbers are not trainable, so we make them as
        # buffer instead of using `nn.Parameter`.
        self.register_buffer(
            'running_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer(
            'running_var', torch.zeros(1, num_features, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (B, C, H, W)

        Returns:
            Tensor of shape (B, C, H, W)
        """
        mean = x.mean(dim=(0, 2, 3), keepdim=True)  # Along (batch, height, width) dimension
        var = torch.mean((x - mean)**2, dim=(0, 2, 3), keepdim=True)

        # Update dataset statistic during training
        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean +\
                self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var +\
                self.momentum * var

        # In training time, we use mini-batch statistic;
        # in testing time, we use the dataset statistic we accumulated.
        if not self.training:
            mean = self.running_mean
            # https://github.com/pytorch/pytorch/issues/1410
            m = x.size(0) * x.size(2) * x.size(3)
            var = m / (m-1) * self.running_var

        x_normed = (x - mean) / (var + self.eps).sqrt_()
        y = self.gamma * x_normed + self.beta

        return y


if __name__ == '__main__':
    bn = MyBatchNorm2d(32)
    x = torch.ones([1, 32, 8, 8])
    y = bn(x)
    print(y.mean())
    bn.eval()
    print(bn(x).mean())
