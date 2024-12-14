import torch as th
import random as rd

from io import TextIOWrapper
from torch.utils.data import Dataset


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