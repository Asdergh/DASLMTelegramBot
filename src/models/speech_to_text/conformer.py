import numpy as np
import torch as th 

from torch.nn import (
    Linear,
    Flatten,
    Conv2d,
    Conv1d,
    GLU,
    Dropout,
    SiLU,
    MaxPool2d,
    Module,
    Sequential,
    BatchNorm2d,
    BatchNorm1d,
    LayerNorm,
    ModuleList,
    Upsample,
    Softmax
)


class ConvSubSampling(Module):

    def __init__(
            self, 
            in_channels: int,
            out_channels: int
    ) -> None:
        
        super().__init__()
        self._net = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=1,
                padding=1,
                kernel_size=3
            ),
            MaxPool2d(kernel_size=3, stride=2),
            BatchNorm2d(num_features=out_channels)
        )

    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net(inputs)


class DepthWiseConv1d(Module):

    def __init__(self, in_channels: int) -> None:

        super().__init__()
        self.in_channels = in_channels
        self._net = ModuleList([
            Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1
            )
            for _ in range(in_channels)
        ])
    
    def __call__(self, inputs: th.Tensor) -> None:
        return th.cat([
            self._net[j](th.unsqueeze(inputs[:, j, :], dim=1))
            for j in range(self.in_channels)
        ], dim=1)


class ConvModule(Module):

    def __init__(
            self, 
            in_features: int, 
            out_channels: int
    ) -> None:
        

        super().__init__()
        self._net = ModuleList([
            LayerNorm(normalized_shape=in_features),
            Sequential(
                Conv1d(
                    in_channels=1,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding=0,
                    stride=1
                ),
                GLU(),
                Upsample(scale_factor=2),
                DepthWiseConv1d(in_channels=out_channels),
                BatchNorm1d(num_features=out_channels),
                SiLU(),
                Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding=0,
                    stride=1
                ),
                Dropout(p=0.45)
            )
        ])
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        norm = th.unsqueeze(self._net[0](inputs), dim=1)
        return self._net[1](norm)


class MultiHeadAttention(Module):

    def __init__(
        self, 
        in_features: int, 
        out_features: int
    ) -> None:

        super().__init__()
        self.d = th.tensor(out_features)

        self.q, self.k, self.v = (
            Linear(in_features=in_features, out_features=out_features) 
            for _ in range(3)
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        q = self.q(inputs)
        k = self.k(inputs)
        v = self.v(inputs)

        dot_norm = th.matmul(v, th.matmul(q.T, k).T) / (th.sqrt(self.d))
        weights = Softmax(dim=1)(dot_norm)
        return weights


class ConformerModule(Module):

    def __init__(self, in_features: int, out_dim: int) -> None:

        super().__init__()
        self.out_dim = out_dim

        self._net = Sequential(
            Linear(in_features=in_features, out_features=32),
            MultiHeadAttention(in_features=32, out_features=64),
            ConvModule(in_features=64, out_channels=128),
        )

    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        
        net = self._net(inputs)
        self.flatten = Flatten()(net)
        self.linear = Linear(in_features=self.flatten.size()[1], out_features=self.ouy_dim)(self.flatten)
        self.out = Softmax(dim=1)(self.linear)

        return self.out
        
        
class PreAttModule(Module):

    def __init__(self, out_features: int) -> None:

        super().__init__()
        self.out_dim = out_features
        self._conv = ConvSubSampling(in_channels=1, out_channels=64)
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        conv = self._conv(inputs)
        self.flatten = Flatten()(conv)
        self.linear = Linear(in_features=self.flatten.size()[1], out_features=128)(self.flatten)
        self.out = Linear(in_features=128, out_features=self.out_dim)(self.linear)

        return self.out
    
class ConformerEncoder(Module):

    def __init__(
        self,
        hiden_dim: int,
        out_dim: int
    ) -> None:
        
        super().__init__()
        self.hiden_dim = hiden_dim
        self.out_dim = out_dim

        self._net = Sequential(
            PreAttModule(out_features=hiden_dim),
            ConformerModule(in_features=hiden_dim, out_dim=out_dim)
        )

    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net(inputs)

    def predict(self, inputs: th.Tensor) -> th.Tensor:
        pass
    
    def save_params(self, filename: str) -> None:
        pass

    def save_to_file(self, filename: str) -> None:
        pass
    



