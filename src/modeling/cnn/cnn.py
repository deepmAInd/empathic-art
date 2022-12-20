import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

from data_preparation import spec_dataset


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class EmotionCNN(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, padding=1, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2), # out: 64 x 254 x 213

            nn.Conv2d(64, 128, kernel_size=5, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=1, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2), # out: 128 x 125 x 104

            nn.Conv2d(128, 256, kernel_size=5, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, padding=1, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2), # out: 256 x 60 x 50

            nn.Dropout(p=.4),

            nn.Flatten(),

            nn.Linear(in_features=256 * 60 * 50, out_features=1024),
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

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            loss = model.train_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        result = evaluate(model, val_loader)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)

    return history

train_num = int(len(spec_dataset) * .7)
val_num = int(len(spec_dataset) * .3)

train_ds, val_ds = random_split(
    dataset=spec_dataset, 
    lengths=[train_num, val_num], 
    generator=torch.Generator().manual_seed(42)
)

batch_size = 64
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

model = EmotionCNN()
history = fit(epochs=1, lr=.001, model=model, train_loader=train_loader, val_loader=val_loader)

# simple_model = nn.Sequential(
#     nn.Conv2d(1, 32, kernel_size=5, padding=1),
#     nn.ReLU(),
#     nn.Conv2d(32, 64, kernel_size=5, padding=1, stride=1),
#     nn.ReLU(),
#     nn.AvgPool2d(2, 2), # out: 64 x 16 x 16
# )

# for images, labels in train_loader:
#     print('images.shape:', images.shape)
#     out = simple_model(images)
#     print('out.shape:', out.shape)
#     break
