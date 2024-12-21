import torch as th
import os
import tqdm as tq

from torch import save
from torchvision.transforms.v2 import Resize
from torchvision.utils import save_image
from torch.utils.data import (
    Dataset,
    DataLoader
)
from torch.nn import (
    Module,
    MSELoss,
    BCELoss
)
from torch.optim import Adam


class FCNSegmentationTrainer:

    def __init__(
        self,
        run_folder: str,
        model: Module,
        train_set: Dataset,
        epochs: int = 100,
        batch_size: int = 32
    ) -> None:
        
        self.run_folder = run_folder
        self.tf_resize = Resize(size=train_set._images_size)
        self.model = model
        self.epochs = epochs
        self.train_loader = DataLoader(dataset=train_set, batch_size=batch_size)
        # self.train_loader = DataLoader(dataset=val_and_test, batch_size=batch_size)

        self.optim = Adam(params=self.model.parameters(), lr=0.01)
        self.loss = MSELoss()
    
    def _save_samples(self, epoch):
        
        gen_path = os.path.join(self.run_folder, "generated_samples")
        epoch_samples = os.path.join(gen_path , f"Epoch_{epoch}_samples")
        paths = [
            gen_path,
            epoch_samples
        ]
        for path in paths:
            if not os.path.exists(path):
                os.mkdir(path)
        
        layers = [layer for (name, layer) in self.model.named_children() if "down" in name]
        out_heatmaps = {}
        x, _ = next(iter(self.train_loader))
        x = x[th.randint(0, 32, size=(1, )), :, :, :]
        for i, layer in enumerate(layers):
            x = layer(x)
            out_heatmaps[i] = self.tf_resize(x)
        
        for out in out_heatmaps.keys():
            
            heatmap = out_heatmaps[out]
            idx = out_heatmaps[out].argsort(dim=1)
            for i in range(heatmap.size()[2]):
                for j in range(heatmap.size()[3]):
                    heatmap[0, :, i, j] = heatmap[0, idx[0, :, i, j], i, j]
            
            heatmap = heatmap[:, :3, :, :]
            save_image(
                tensor=heatmap,
                fp=os.path.join(epoch_samples, f"Layer_{out}_generated_heatmap.png")
            )
    

    def _train_on_epoch(self, epoch: int) -> float:

        local_loss = 0.0
        for (image, seg_masks) in self.train_loader:
            
            self.optim.zero_grad()
            out = self.model(image)
            loss = 0.0
            for (mask_pred, mask_target) in zip(out, seg_masks):
                loss += self.loss(mask_target, mask_pred)
            
            loss.backward()
            self.optim.step()
            local_loss += loss
        
        return local_loss

    def train(self) -> dict[list]:

        loss = []
        for epoch in tq.tqdm(range(self.epochs), colour="green"):
            
            local_loss = self._train_on_epoch(epoch=epoch)
            loss.append(local_loss)
            self._save_samples(epoch=epoch)

            print(f"Epoch: [{epoch}], Loss: [{local_loss}]")
        
        save(
            obj=self.model.stated_dict(),
            f=os.path.join(self.run_folder, "model_params.pt")
        )
        return {
            "loss": loss
        }
            
                
            