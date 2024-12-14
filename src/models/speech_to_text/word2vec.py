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
    Softmax
)
from torch.utils.data import (
    Dataset,
    DataLoader
)


class Word2VecRec(Module):

    def __init__(self, tokens_n: int, embedding_dim: int) -> None:

        super().__init__()

        self.embedding = Embedding(num_embeddings=tokens_n, embedding_dim=embedding_dim)
        self.lstm = LSTM(
            input_size=embedding_dim,
            hidden_size=128,
            num_layers=3
        )
        self.linear = Linear(in_features=128, out_features=embedding_dim)
        
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        em = self.embedding(inputs)
        lstm = self.lstm(em)
        projection = self.linear(lstm)

        return Softmax(dim=1)(projection)

    def save_weights(self, filename: str) -> None:
        th.save(self.state_dict, filename)
    
    def load_weights(self, filename: str) -> None:
        self.load_state_dict(th.load(filename, weights_only=True))

    def predict(self, inputs: th.Tensor.long) -> th.Tensor:
        return self.linear(inputs)



class Word2VecRnnTrainer:

    def __init__(
        self,
        epochs: int,
        batch_size: int,
        data_source: str | th.utils.data.Dataset 
    ) -> None:
        
        super().__init__()



