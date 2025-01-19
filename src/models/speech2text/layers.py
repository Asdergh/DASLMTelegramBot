import torch as th
from math import log2
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
    Softmax,
    ConvTranspose1d,
    ReLU,
    Tanh,
    LSTM,
    Embedding
)


__all__ = [
    "Embedding",
    "log2",
    "Linear",
    "Flatten",
    "Conv2d",
    "Conv1d",
    "GLU",
    "Dropout",
    "SiLU",
    "MaxPool2d",
    "Module",
    "Sequential",
    "BatchNorm2d",
    "BatchNorm1d",
    "LayerNorm",
    "ModuleList",
    "Upsample",
    "Softmax",
    "ConvTranspose1d",
    "ReLU",
    "Tanh",
    "ConvTransposeSubSampling1d",
    "ConvSubSampling",
    "DepthWiseConv1d",
    "ConvModule",
    "MultiHeadAttention",
    "LSTM"
]


class ConvTransposeSubSampling1d(Module):

    def __init__(
            self, 
            in_channels: int,
            out_channels: int
    ) -> None:
        
        super().__init__()
        self._net_ = Sequential(
            ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                padding=0,
                kernel_size=2
            ),
            BatchNorm1d(num_features=out_channels),
            ReLU()
        )

    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net_(inputs)
    
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
            out_channels: int,
            from_vector: bool = False,
            in_channels: int = None,
            in_features: int = None
    ) -> None:
        

        super().__init__()
        assert not(in_channels is None and in_features is None), "You must use :[in_channels: int] or [in_features: int] as an input arg"
            
        if in_channels is None:
            in_channels = 1
        

        self.fv = from_vector
        self._conv_ = Sequential(
                Conv1d(
                    in_channels=in_channels,
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
        
        self._net_ = self._conv_
        if from_vector:
            self._net_ = ModuleList([
                LayerNorm(normalized_shape=in_features),
                self._conv_
            ])
        
    
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        if self.fv:
            norm = th.unsqueeze(self._net_[0](inputs), dim=1)
            return self._net_[1](norm)

        return self._net_(inputs)


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