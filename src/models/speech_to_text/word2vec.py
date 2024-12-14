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



class TextSet(Dataset):

    def __init__(
        self,
        data_source: str | TextIOWrapper,
        sequences_lenght: int,
        replace_order: dict = None,
        separation_order: int = 3
        
    ) -> None:
        
        super().__init__()
        self.text = data_source
        self.sequences_lenght = sequences_lenght
        self.sep_order = separation_order

        if isinstance(data_source, TextIOWrapper):
            self.text = data_source.read()
            data_source.close()
        
        ro = replace_order
        if ro is None:
            ro = {
                ",": " ",
                ".": " ",
                "-": " ",
                "\n": "@"
            }

        self.text = self.text.lower()
        for rep_item in ro.keys():
            self.text = self.text.replace(rep_item, ro[rep_item])
        
        
        self.word_to_idx = {word.replace("@", ""): i for (i, word) in enumerate(set(self.text.split()))}
        self.idx_to_word = {i: word.replace("@", "") for (word, i) in self.word_to_idx.items()}
        
        self.text = self.text.split("@")
        self._string_separator()
        self._text_trashhold()
        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    def __len__(self) -> int:
        return len(self.text)


    def __getitem__(self, idx) -> tuple[th.Tensor, th.tensor]:

        string = self.text[idx].split()
        
        if len(string) > self.sequences_lenght:
            
            idx_i = rd.randint(0, (len(string) - self.sequences_lenght))
            label = string[idx_i + self.sequences_lenght - 1]    
            string = string[idx_i: idx_i + self.sequences_lenght]
        
        else:
            label = self.text[idx].split()[0]
        
        out_sample = []
        for token in string:
            self._expand_vocs(token)
            out_sample.append(self.word_to_idx[token])
        
        self._expand_vocs(label)
        return (th.Tensor(out_sample), th.tensor(self.word_to_idx[label]))


    def _text_trashhold(self):

        _tmp_text = []
        for sample in self.text:
            if len(sample.split()) >= self.sequences_lenght:
                _tmp_text.append(sample)

        self.text = _tmp_text


    def _string_separator(self):

        _tmp_text = []
        for sample in self.text:

            if len(sample.split()) >= self.sequences_lenght * self.sep_order:
                for i in range(self.sep_order):
                    try:
                        string = sample.split()[i * self.sequences_lenght: (i + 1) * self.sequences_lenght]
                        string = " ".join(token for token in string)
                        _tmp_text.append(string)
                    
                    except BaseException:
                        pass

            else:
                _tmp_text.append(sample)
        
        self.text = _tmp_text
    

    def _expand_vocs(self, token):

        if not token in self.word_to_idx:

            self.word_to_idx[token] = len(self.word_to_idx) + 1
            self.idx_to_word[self.word_to_idx[token]] = token
    
    def sequences_to_texts(self, inputs: th.Tensor) -> list[str]:

        return [
            " ".join(self.idx_to_word[idx] for idx in sample)
            for sample in inputs.tolist()
        ]
         

        
class Word2VecRnnTrainer:

    def __init__(
        self,
        epochs: int,
        batch_size: int,
        data_source: str | th.utils.data.Dataset 
    ) -> None:
        
        super().__init__()


if __name__ == "__main__":
    
    txt_file = open("C:\\Users\\1\\Desktop\\res_mat.txt", "r")
    test_set = TextSet(data_source=txt_file, sequences_lenght=45)
    test_loader = DataLoader(dataset=test_set, batch_size=32)

    # print(len(test_loader))
    samples_txt = []
    samples_labels = []
    # print(len(test_loader))
    for sample in test_loader:
        samples_txt.append(sample[0])
        samples_labels.append(sample[1])
    
    res_txt = th.cat(samples_txt, dim=0)
    res_labels = th.cat(samples_labels, dim=0)
    print(res_txt.size(), res_labels.size())
    txt_from_tensor = test_set.sequences_to_texts(res_txt)
    for string in txt_from_tensor:
        print(string)
    # print(samples)
    # res = th.Tensor(samples)
    # print(samples.size())
