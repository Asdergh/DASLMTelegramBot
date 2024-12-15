# TODO write word2vec trainer
# TODO end up with TextSet cll

import torch as th
import random as rd
import tqdm as tq

from io import TextIOWrapper
from torch.nn import (
    LSTM,
    Embedding,
    Linear,
    Module,
    Softmax,
    ModuleList,
    Sequential,
    Parameter
)
from torch.utils.data import (
    Dataset,
    DataLoader
)


class SeqDecoder(Module):

    def __init__(
            self, 
            tokens_n: int, 
            embedding_dim: int, 
            max_sequence_length: int = 20
    ) -> None:

        super().__init__()
        self._rnn_base = Sequential(
            Embedding(num_embeddings=tokens_n, embedding_dim=embedding_dim),
            LSTM(
                input_size=embedding_dim,
                hidden_size=128,
                num_layers=3
            )
        )

        self._linear_base = Sequential(
            Linear(in_features=128, out_features=embedding_dim),
            Linear(in_features=embedding_dim, out_features=tokens_n),
            Softmax(dim=1)
        )
        
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        
        rnn_out, _ = self._rnn_base(inputs)
        return self._linear_base(rnn_out)
        




