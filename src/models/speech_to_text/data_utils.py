import torch as th
import random as rd
import os
import pandas as pd

from io import TextIOWrapper
from torch.utils.data import Dataset
from torchaudio import load
from torchaudio.transforms import MelSpectrogram



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
    


class S2TDataset(Dataset):


    def __init__(
        self,
        data_dir: str,
        transforms: th.nn.Module,
        replace_order: dict = None,
        sequences_lenght: int = 20
    ) -> None:

        super().__init__()
        self.seq_len = sequences_lenght
        self.ro = replace_order
        if self.ro is None:
            self.ro = {
                ",": " ",
                ".": " ",
                "-": " ",
                "\n": "@"
            }

        self.mels, self.texts = [], []
        self.word_to_idx = {}
        self.idx_to_word = {}

        for split in os.listdir(data_dir):
            split_path = os.path.join(data_dir, split)
            for batch in os.listdir(split_path):
                
                batch_path = os.path.join(split_path, batch)
                annots_path = f"{split}-{batch}.trans.txt"
                print(os.path.join(batch_path, annots_path))
                self._tabulate_annots(os.path.join(batch_path, annots_path))

                tmp_df = pd.read_csv(annots_path, sep="\t")
                txt_samples, track_ids = self._txt_mel_aug(data_frame=tmp_df)
                
                self.texts += txt_samples
                self.mels += [transforms(load(os.path.join(batch_path, f"{track_id}.flac")) 
                                  for track_id in track_ids)]
    

    def __len__(self) -> int:
        return len(self.mels)
    
    def __getitem__(self, idx) -> tuple[th.Tensor]:
        
        txt_sample = th.Tensor([self.word_to_idx[token] for token in self.texts[idx]])
        return (
            self.mels[idx],
            txt_sample
        )
        
                
    def _txt_mel_aug(self, data_frame) -> list[str]:


        texts = data_frame["TEXT_CONTENT"].to_list()
        tracks = data_frame["TRACK_IDS"].to_list()
        out_txt, out_mels = []

        for (string, track_id) in zip(texts, tracks):
            
            if len(string.split()) >= self.seq_len:

                for (rep_sim, new_sim) in enumerate(self.ro):
                    string.replace(rep_sim, new_sim)

                out_mels.append(track_id)
                out_txt.append(string)
                for token in string.split():
                    if not token in self.word_to_idx:

                        self.word_to_idx[token] = len(self.word_to_idx) + 1
                        self.idx_to_word[self.word_to_idx[token]] = token
        
        return (out_txt, out_mels)

            


            
    def _tabulate_annots(self, filename: str, outfile: str = None) -> None:

        str_buffer = []
        print(filename)
        with open(filename, "r") as txt_file:

            text = txt_file.readlines()
            for string in text:

                buffer = string.split()
                track_idx = buffer[0]
                txt_content = " ".join(token for token in buffer[1:])

                str_buffer += [track_idx + "\t" + txt_content.lower(), ]
        
        str_buffer = "TRACK_IDS\tTEXT_CONTENT\n" + "\n".join(string for string in str_buffer)
        out_file = filename
        if outfile is not None:
            out_file = outfile

        with open(out_file, "w") as txt_file:
            txt_file.write(str_buffer)
                



if __name__ == "__main__":

    transforms = MelSpectrogram()
    data_dir = "C:\\Users\\1\\Desktop\\dev-clean\\LibriSpeech\\dev-clean"
    train_set = S2TDataset(data_dir=data_dir, transforms=transforms)
