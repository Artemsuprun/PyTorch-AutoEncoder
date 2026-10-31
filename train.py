#
# File:   train.py
# Author: Artem Suprun
# Date:   10/29/2026
# Description: Training pipeline for a generic autoencoder.
#

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os # for making sure the checkpoint file exists

from dataset import AutoEncoderDataset
from model import AutoEncoder


class Trainer:

    def __init__(self, model, optimizer, criterion, device=None):
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def train_epoch(self, dataloader):
        self.model.train()

        total_loss = 0.0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()

        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def fit(self, train_loader, val_loader, epochs, checkpoint_path):
        best_loss = float("inf")

        for epoch in range(epochs):

            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            print(f"Epoch [{epoch + 1}/{epochs}] Train Loss: {train_loss:.6f} Validation Loss: {val_loss:.6f}")

            # Save the best model
            if val_loss < best_loss:
                best_loss = val_loss

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": best_loss,
                        "config": {
                            "encoder_dim": [784, 128, 64],
                            "activation": "relu",
                            "output_activation": "sigmoid"
                        }
                    },
                    checkpoint_path
                )

                print("Saved new best checkpoint.")


def main():

    # Configuration
    batch_size = 64
    epochs = 20
    learning_rate = 0.001
    checkpoint_path = "checkpoints/autoencoder.pth"

    # make sure the directory exists
    os.makedirs("checkpoints", exist_ok=True)

    # Dataset transformations
    transform = transforms.ToTensor()

    corruption = transforms.Compose([
        transforms.RandomApply([
            transforms.GaussianBlur(
                kernel_size=5,
                sigma=(0.5, 2.0)
            )
        ]),
        #transforms.RandomApply([
        #    transforms.RandomErasing(
        #        p=1.0,
        #        scale=(0.02, 0.15)
        #    )
        #])
    ])

    train_dataset = AutoEncoderDataset(
        datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transform
        ),
        corruption=corruption
    )

    val_dataset = AutoEncoderDataset(
        datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=transform
        ),
        corruption=corruption
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Initialize model
    model = AutoEncoder(
        encoder_dim=[784, 128, 64],
        activation="relu",
        output_activation="sigmoid"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    criterion = nn.MSELoss()

    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_path=checkpoint_path
    )


if __name__ == "__main__":
    main()
