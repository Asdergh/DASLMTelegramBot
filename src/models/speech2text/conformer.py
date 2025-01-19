import numpy as np
import torch as th 
from layers import *



class ConformerModule(Module):

    def __init__(
        self,
        in_features: int,
        encoding_dim: int,
        out_channels: int = 64,
        att_features: int = 200
    ) -> None:

        super().__init__()
        self._linear_ = Linear(in_features=(out_channels * att_features), out_features=encoding_dim)
        self._flatten_ = Flatten()
        self._act_ = Softmax(dim=1)
        self._net = Sequential(
            Linear(in_features=in_features, out_features=32),
            MultiHeadAttention(in_features=32, out_features=att_features),
            ConvModule(in_features=att_features, out_channels=out_channels),
        )

    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        
        net = self._net(inputs)
        flatten = self._flatten_(net)
        linear = self._linear_(flatten)
        return self._act_(linear)

        
        
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
        encoding_dim: int
    ) -> None:
        
        super().__init__()
        self.hiden_dim = hiden_dim
        self.encoding_dim = encoding_dim

        self._net = Sequential(
            PreAttModule(out_features=hiden_dim),
            ConformerModule(in_features=hiden_dim, encoding_dim=encoding_dim)
        )

    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net(inputs)


class ConformerDecoder(Module):

    def __init__(
        self,
        encoding_dim: int,
        out_channels: int,
        target_size: int = 200,
        patch_size: int = 100
    ) -> None:
        
        super().__init__()
        conv_n = int(log2(target_size)) - int(log2(patch_size))
        self._linear_ = Linear(in_features=encoding_dim, out_features=patch_size)
        self._net_ = Sequential(
            ConvTranspose1d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0
            ), *[
            ConvTransposeSubSampling1d(in_channels=out_channels, out_channels=out_channels)
            for _ in range(conv_n - 1)
        ], Tanh())
    
    def __call__(self, inputs: th.Tensor):
        
        lin = self._linear_(inputs).unsqueeze(dim=1)
        return self._net_(lin)
        
    

A = th.normal(0.12, 1.12, (32, 1, 45, 200))
encoder = ConformerEncoder(
    hiden_dim=128,
    encoding_dim=64
)

decoder = ConformerDecoder(
    encoding_dim=64,
    out_channels=45,
    target_size=128,
    patch_size=64
)
print(decoder(encoder(A)).size())

# class ConformerDecoder(Module):

#     def __init__(
#         self,
        
#     ):
#         super().__init__()
    

