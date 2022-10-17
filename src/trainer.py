from typing import List, Optional, Tuple
from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset

from metrics import *


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: Optional[Dataset],
        test_loader: Optional[Dataset],
        criterion: torch.nn.Module,
        optimizer: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR],
        epochs: Optional[int] = 5,
        metrics: List[Metric] = [],
        device: str = "cpu",
    ) -> None:
        """
        Initialize the trainer class
        :param model: model to be trained
        :param device: device to be used for training
        :param train_loader: train data loader
        :param test_loader: test data loader
        :param criterion: loss function
        :param optimizer: optimizer
        :param epochs: number of epochs
        :param metrics: list of metrics to be used for evaluation
        """
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.metrics = metrics

    def train(self):
        """
        Train the model
        """
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            pbar = tqdm(self.train_loader)
            train_loss: float = 0.0
            for step, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                train_loss 	+= loss.item()
                loss.backward()
                self.optimizer.step()
                metrics_str = " ".join(
                    [
                        f"{metric.name}={metric(output, target):.4f}"
                        for metric in self.metrics
                    ]
                )
                pbar.set_description(
                    desc=f"Epoch {epoch}: loss={train_loss/(step + 1):.4f} {metrics_str}"
                )
            self.evaluate()

    def evaluate(self):
        """
        Evaluate the model on test set
        """
        self.model.eval()
        with torch.no_grad():
            preds: torch.Tensor = None
            label_ids: torch.Tensor = None
            pbar = tqdm(self.test_loader)
            val_loss: float = 0.0
            for step, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss = self.criterion(output, target).item()
                if preds is None:
                    preds = output.detach()
                else:
                    preds = torch.cat((preds, output.detach()), dim=0)
                if label_ids is None:
                    label_ids = target.detach()
                else:
                    label_ids = torch.cat((label_ids, target.detach()), dim=0)
                metrics_str = " ".join(
                    [
                        f"{metric.name}={metric(preds, label_ids):.4f}"
                        for metric in self.metrics
                    ]
                )
                pbar.set_description(
                    desc=f"Evaluate: loss={val_loss/(step + 1):.4f} {metrics_str}"
                )
