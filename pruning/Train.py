from pathlib import Path

print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())
import torch
import math
from tqdm.auto import tqdm


class Trainer:
    def __init__(
            self,
            model,
            epochs,
            train_loader,
            criterion,
            optimizer,

    ):
        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss = {"train": []}
        self.acc = []

    def train_step(self):
        self.model.train()  # setting model to training mode
        train_loss = 0
<<<<<<< HEAD
        # device=("cuda:3" if torch.cuda.is_available() else "cpu")
        device = "cpu"
=======
        device=("cuda" if torch.cuda.is_available() else "cpu")
>>>>>>> 2e0c0079159a7fff9c165d25f5438e813114f034
        self.model.to(device)

        # setting requires_grad =True
        with torch.set_grad_enabled(True):
            for batch, (X, y) in enumerate(self.train_loader):
                # forward pass
                X,y= X.to(device),y.to(device)
                y_pred = self.model(X)
                # calculate and sum loss
                loss = self.criterion(y_pred, y)
                train_loss += loss.item()
                # backward pass and setting previous epoch gradients to zero
                self.optimizer.zero_grad()
                loss.backward()
                # parameters update
                self.optimizer.step()
                # self.scheduler.step()
        return train_loss

    def train(self):
        # training
        for epoch in tqdm(range(self.epochs)):
            train_loss = self.train_step()
            print(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {math.log(train_loss):.4f} | "
            )
            self.loss["train"].append(math.log(train_loss))  # using log loss

        return self.model
