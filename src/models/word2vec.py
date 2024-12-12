# TODO write word2vec trainer
# TODO end up with TextSet cll

import torch as th
import random as rd

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



class TextSet(Dataset):

    def __init__(
        self,
        data_source: str | TextIOWrapper,
        sequences_lenght: int
        
    ) -> None:
        
        super().__init__()
        self.text = data_source
        self.sequences_lenght = sequences_lenght

        if isinstance(data_source, str):
            self.text = data_source.read()
            data_source.close()
        
        self.word_to_idx = {word: i for (i, word) in enumerate(set(self.text.replace("\n", "@").split()))}
        self.idx_to_word = {i: word for (word, i) in self.word_to_idx.items()}
        self.text = self.text.split("@")
    
    def __getitem__(self, idx):

        row = self.text[idx].split()
        if len(row) > self.sequences_lenght:
            row = row[:self.sequence_lenght]

        elif len(row) < self.sequences_lenght:

            add = self.text[idx + 1].split()[:self.sequences_lenght - len(row)]
            row += add

        for  

    


        


            
        
class Word2VecRnnTrainer:

    def __init__(
        self,
        epochs: int,
        batch_size: int,
        data_source: str | th.utils.data.Dataset 
    ) -> None:
        
        super().__init__()





