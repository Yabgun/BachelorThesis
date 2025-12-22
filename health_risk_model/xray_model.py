from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models


class XrayRiskModel:
    def __init__(self, num_classes: int = 2, pretrained: bool = True, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        self.model.to(self.device)

    def fit(
        self,
        train_loader: DataLoader,
        num_epochs: int = 5,
        lr: float = 1e-4,
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for _ in range(num_epochs):
            for images, labels, _ in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def evaluate(self, data_loader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels, _ in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0

    def predict_proba(
        self,
        data_loader: DataLoader,
    ) -> Tuple[List[str], List[int], List[float]]:
        self.model.eval()
        paths: List[str] = []
        labels: List[int] = []
        probs: List[float] = []

        softmax = nn.Softmax(dim=1)

        with torch.no_grad():
            for images, batch_labels, batch_paths in data_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                prob_matrix = softmax(outputs)
                for i in range(prob_matrix.size(0)):
                    paths.append(batch_paths[i])
                    labels.append(int(batch_labels[i]))
                    probs.append(float(prob_matrix[i, 1]))

        return paths, labels, probs

