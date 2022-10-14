import pandas as pd

import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchaudio.transforms as T


from torch.utils.data.dataloader import DataLoader, Dataset, T_co


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class EmotionCNN(nn.Module):    

    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            # first `layer`
            nn.Conv2d(3, 32, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, padding=1, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2), # out: 64 x 16 x 16

            # second `layer`
            nn.Conv2d(64, 128, kernel_size=5, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=1, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2), # out: 128 x 8 x 8

            # third `layer`
            nn.Conv2d(128, 256, kernel_size=5, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, padding=1, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2), # out: 256 x 4 x 4

            nn.Dropout(p=.4),

            nn.Flatten(),

            nn.Linear(in_features=256 * 4 * 4, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10),
        )

    def forward(self, xb: torch.Tensor):
        return self. network(xb)

    def train_step(self, batch):
        images, labels = batch
        out = self(images) # generate predictions
        loss = F.cross_entropy(out, labels) # calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images) # generate predictgions
        loss = F.cross_entropy(out, labels) # calculate loss
        acc = accuracy(out, labels) # calculate accuracy
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()

        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch {epoch}, train_loss: {result['train_loss']}, \
        val_loss: {result['val_loss']}, val_acc: {result['val_acc']}")


@torch.no_grad()
def evaluate(model: EmotionCNN, val_loader: DataLoader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model: EmotionCNN, train_loader, val_loader, opt_func = optim.Adam):
    history = []
    optimizer = opt_func(model.parameters(), lr=lr)

    for epoch in epochs:
        model.train()
        train_losses = []

        for batch in train_loader:
            loss = model.train_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        result = evaluate(epoch, result)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)

    return 


model = EmotionCNN()
history = fit(epochs=10, lr=.001, model=model, train_loader=train_loader, val_loader=val_loader)


