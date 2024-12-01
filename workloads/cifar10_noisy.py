import torch
import random
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor, RandomHorizontalFlip, RandomRotation, RandomAffine, ColorJitter

def add_label_noise(batch, noise_ratio):
    """Add noise to labels between similar classes for CIFAR-10."""
    labels = batch["label"]
    noisy_labels = labels.clone()
    
    # Define the mapping for similar classes
    similar_classes = {
        9: 1,  # truck -> automobile
        2: 0,  # bird -> airplane
        4: 7,  # deer -> horse
        3: 5,  # cat -> dog
    }
    
    # Select indices to corrupt
    num_noisy = int(len(labels) * noise_ratio)
    noisy_indices = torch.randperm(len(labels))[:num_noisy]
    
    # Replace selected labels with their similar counterparts
    for idx in noisy_indices:
        original_label = labels[idx].item()
        if original_label in similar_classes:
            noisy_labels[idx] = similar_classes[original_label]
    
    batch["label"] = noisy_labels
    return batch

def load_dataset(partitioners, batch_size=64, test_size=0.1, noise_ratio=0.0):

    fds = FederatedDataset(
        dataset="cifar10",
        partitioners={"train": partitioners},
    )

    def apply_transforms(batch):

        batch["img"] = [
            Compose(
                [
                    RandomHorizontalFlip(),
                    RandomRotation(10),
                    RandomAffine(0, shear=10, scale=(0.8,1.2)),
                    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    ToTensor(),
                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                ]
            )(img)
            for img in batch["img"]
        ]
        return add_label_noise(batch, noise_ratio)

    def apply_test_transforms(batch):

        batch["img"] = [
            Compose(
                [
                    ToTensor(),
                    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                ]
            )(img)
            for img in batch["img"]
        ]
        return batch

    testloader = DataLoader(
        fds.load_split("test").with_transform(apply_test_transforms), batch_size=batch_size
    )

    def get_client_loader(cid: str):
        client_dataset = fds.load_partition(int(cid), "train")
        client_dataset_splits = client_dataset.train_test_split(
            test_size=test_size, seed=42
        )
        trainset = client_dataset_splits["train"]
        valset = client_dataset_splits["test"]
        trainloader = DataLoader(
            trainset.with_transform(lambda batch: apply_transforms(batch)), batch_size=batch_size
        )
        valloader = DataLoader(
            valset.with_transform(apply_test_transforms), batch_size=batch_size
        )
        return trainloader, valloader

    return testloader, get_client_loader
