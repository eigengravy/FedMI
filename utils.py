
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr_datasets import FederatedDataset
from scipy.stats import pearsonr, kendalltau
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor

from typing import List

import torch
import torch.nn as nn


def federated_averaging(
    global_model: nn.Module, models: List[nn.Module], device
) -> nn.Module:
    global_model.to(device)
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack(
            [client_model.state_dict()[k].float() for client_model in models], 0
        ).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model


def evaluate(model: nn.Module, test_loader: DataLoader, device, process_batch) -> Tuple[float, float]:
    model.eval().to(device)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            images, labels = process_batch(batch)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += nn.functional.cross_entropy(
                outputs, labels, reduction="sum"
            ).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy

