import torch as th
import tqdm as tq
from torch.nn import (
    MSELoss,
    CrossEntropyLoss,
    Module,
    L1Loss
)
from torch.optim import (
    SGD,
    Adam
)
from torch.utils.data import DataLoader


class SeqDecoderTrain:

    def __init__(
        self,
        loader: DataLoader,
        model: Module,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> None:
        
        self.loader = loader
        self.epochs = epochs
        self.batch_size = batch_size

        self._model = model
        self.optim = SGD(self._model.parameters(), lr=0.01)
        self.loss = L1Loss()
    
    def _levenstain_distance(self, string0, string1) -> int:
        
        n = len(string0)
        m = len(string1)
        d_map = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

        for i in range(n + 1):
            d_map[i][0] = i

        for j in range(m + 1):
            d_map[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):

                if string0[i - 1] == string1[j - 1]:
                    d_map[i][j] = d_map[i - 1][j - 1]
                
                else:
                    d_map[i][j] = 1 + min(d_map[i - 1][j], d_map[i][j - 1], d_map[i - 1][j - 1])
        
        
        return d_map[n][m]
    

    def _train_on_epoch(self, epoch: int) -> float:

        result_loss = 0.
        for i, (_, tokens, _) in enumerate(tq.tqdm(self.loader)):

            self.optim.zero_grad()

            out = self._model(tokens.to(th.long))
            loss = self.loss(th.argmax(out, dim=1), tokens)
            result_loss += loss.item()
            loss.backward()

            self.optim.step()
            if i == len(self.loader.dataset) - 1:
                
                result_loss = result_loss / len(self.loader.dataset)
                return result_loss
    
    def train(self):

        loss_history = []
        for epoch in range(self.epochs):

            loss = self._train_on_epoch(epoch=epoch)
            loss_history.append(loss)







                

                

            
            
            
            

    
    
        