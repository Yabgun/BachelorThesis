import os
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class XrayImageDataset(Dataset):
    def __init__(self, root_dir: str, img_size: int = 224):
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.dataset = datasets.ImageFolder(root=root_dir, transform=self.transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        path, _ = self.dataset.samples[idx]
        return image, label, path

    @property
    def classes(self) -> List[str]:
        return self.dataset.classes


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    train_dataset = XrayImageDataset(train_dir, img_size=img_size)
    test_dataset = XrayImageDataset(test_dir, img_size=img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    class_names = train_dataset.classes
    return train_loader, test_loader, class_names

