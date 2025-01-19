import torch as th
from layers import *

class SequenceTranslationNet(Module):


    def __init__(
        self,
        seq_len: int,
        time_dim: int,
        vocab_size: int,
        embedding_dim: int,
        rnn_features: int,
        out_channels: int
    ) -> None:
        
        super().__init__()
        self._linear_ = Linear(
            rnn_features, 
            time_dim
        )
        self._embedding_ = Embedding(
            vocab_size,
            embedding_dim
        )
        self._rnn_ = LSTM(
            embedding_dim,
            rnn_features,
            num_layers=3
        )
        self._conv_ = ConvModule(
            from_vector=False,
            in_channels=seq_len,
            out_channels=out_channels
        )
    
    def __call__(self, inputs: th.Tensor) -> None:

        embedding = self._embedding_(inputs)
        rnn, _ = self._rnn_(embedding)
        linear = self._linear_(rnn)
        return self._conv_(linear)


if __name__ == "__main__":

    seq_len = 34
    time_dim = 200
    tokens = th.randint(0, 100, (32, seq_len))
    model = SequenceTranslationNet(
        seq_len=seq_len,
        time_dim=time_dim,
        vocab_size=100,
        embedding_dim=200,
        rnn_features=128,
        out_channels=64
    )
    print(model(tokens).size())


        





