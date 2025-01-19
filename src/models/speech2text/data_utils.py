import torch as th
import random as rd
import tqdm as tq
import os
import pandas as pd
import matplotlib.pyplot as plt

from io import TextIOWrapper
from torch.utils.data import (
    Dataset, 
    DataLoader
)
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
        sequences_lenght: int = 20,
        mel_time_range: int = 300
    ) -> None:

        super().__init__()
        self.tf = transforms
        self.seq_len = sequences_lenght
        self.ro = replace_order
        self.mel_time_range = mel_time_range

        if self.ro is None:
            self.ro = {
                ",": " ",
                ".": " ",
                "-": " ",
                "\n": "@"
            }

        self.track_ids, self.texts = [], []
        self.word_to_idx = {}
        self.idx_to_word = {}

        for split in os.listdir(data_dir):
            split_path = os.path.join(data_dir, split)
            for batch in os.listdir(split_path):
                
                batch_path = os.path.join(split_path, batch)
                annots_path = os.path.join(batch_path, f"{split}-{batch}.trans.txt")
                print(annots_path)

                self._tabulate_annots(annots_path)
                txt_samples, track_ids = self._txt_mel_aug(filename=annots_path)
                
                self.texts += txt_samples
                self.track_ids += [os.path.join(batch_path, f"{track_id}.flac") 
                                  for track_id in track_ids]


    def _txt_mel_aug(self, filename: str) -> list[str]:


        with open(filename, "r") as txt_annots:
            data = txt_annots.readlines()

        texts = [row.split("\t")[1] for row in data]
        tracks = [row.split("\t")[0] for row in data]
        out_txt, out_mels = [], []

        for (string, track_id) in zip(texts, tracks):
            
            if len(string.split()) >= self.seq_len:

                for (rep_sim, new_sim) in self.ro.items():
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
        with open(filename, "r") as txt_file:

            text = txt_file.readlines()
            for string in text:

                buffer = string.split()
                track_idx = buffer[0]
                txt_content = " ".join(token for token in buffer[1:])

                str_buffer += [track_idx + "\t" + txt_content.lower(), ]
        
        start_token = "TRACK_IDS\tTEXT_CONTENT\n"
        if str_buffer[0] == start_token.lower():
            start_token = ""

        str_buffer =  start_token + "\n".join(string for string in str_buffer)
        out_file = filename
        if outfile is not None:
            out_file = outfile

        with open(out_file, "w") as txt_file:
            txt_file.write(str_buffer)
    
    
    def _tensor_to_str(self, input: th.Tensor.long) -> str:
        return " ".join(
            self.idx_to_word[idx] 
            for idx in input.tolist()
        )

    def sequences_to_text(self, inputs: th.Tensor) -> str | list[str]:

        if len(inputs.size()) == 2:
            texts = []
            for sample in inputs:
                texts.append(self._tensor_to_str(sample))
            return texts
        
        return self._tensor_to_str(inputs)


            
    def __len__(self) -> int:
        return len(self.track_ids)
    

    def __getitem__(self, idx) -> tuple[
        th.Tensor,
        th.Tensor.long,
        th.tensor
    ]:
        
        wave, _ = load(self.track_ids[idx])
        mel_sample = self.tf(wave)

        txt_sample = self.texts[idx].split()
        label = txt_sample[-1]
        if idx < len(self.texts) - 1:
            label = self.texts[idx + 1].split()[0]
        
        if len(txt_sample) > self.seq_len:

            rand_idx = rd.randint(0, len(txt_sample) - self.seq_len)
            label = txt_sample[rand_idx + self.seq_len - 1]
            txt_sample = txt_sample[rand_idx: rand_idx + self.seq_len]

        return (
            mel_sample[:, :, :self.mel_time_range],
            th.Tensor([self.word_to_idx[token] for token in txt_sample]),
            th.tensor(self.word_to_idx[label])
        )
                



if __name__ == "__main__":

    transforms = MelSpectrogram()
    data_dir = "C:\\Users\\1\\Desktop\\dev-clean\\LibriSpeech\\dev-clean"
    train_set = S2TDataset(data_dir=data_dir, transforms=transforms)
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
    
    test_mels, test_tokens = [], []
    for i, (mel, tokens, _) in enumerate(tq.tqdm(train_loader, colour="GREEN")):
        
        test_mels.append(mel), test_tokens.append(tokens)
        print(mel.size(), tokens.size())

    
    test_mels = th.cat(test_mels, dim=0)
    for (sample, sample_tf) in zip(train_set.sequences_to_text(test_tokens[:][0]), test_tokens[:][0]):
        print(sample, sample_tf.tolist(), sep="\t\t")
    
    plt.style.use("dark_background")
    _, axis = plt.subplots(nrows=3)
    for i in range(axis.shape[0]):

        idx = rd.randint(0, 100)
        axis[i].imshow(test_mels[idx][0], cmap="jet")
    
    plt.show()