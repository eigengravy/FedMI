import argparse
import torch
import torch.nn as nn
from torch import optim
import wandb
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from utils import federated_averaging, evaluate
from models.simple_cnn import SimpleCNN
import random
import numpy as np
from tqdm import tqdm
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from losses import criterions

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_clients', type=int, default=100)
parser.add_argument('--num_rounds', type=int, default=100)
parser.add_argument('--local_epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--participation_fraction', type=float, default=0.1)
parser.add_argument('--partitioner', type=str, choices=["iid", "dirichlet"], default="iid")
parser.add_argument('--partition_alpha', type=float, default=0.5)
parser.add_argument('--loss', type=str, choices=criterions.keys())
parser.add_argument('--dataset', type=str, choices=["cifar10", "cifar100", "cifar10_noisy"], default="cifar10")
parser.add_argument('--noise_ratio', type=float, default=0.0)

args = parser.parse_args()

seed = args.seed
num_clients = args.num_clients
num_rounds = args.num_rounds
local_epochs = args.local_epochs
batch_size = args.batch_size
participation_fraction = args.participation_fraction
partitioner_type = args.partitioner
partition_alpha = args.partition_alpha
loss_type = args.loss
noise_ratio = args.noise_ratio

if args.dataset == "cifar10":
    from workloads.cifar10 import load_dataset, process_batch
    num_classes = 10
    partition_by = "label"
elif args.dataset == "cifar100":
    from workloads.cifar100 import load_dataset, process_batch
    num_classes = 100
    partition_by = "fine_label"
elif args.dataset == "cifar10_noisy":
    from workloads.cifar10_noisy import load_dataset, process_batch
    num_classes = 10
    partition_by = "label"
    
DEVICE_ARG = f"cuda:{args.device}"
DEVICE = torch.device(DEVICE_ARG if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

wandb.login()

wandb.init(
    project=f"fedmi",
    group=f"{args.dataset}-{args.loss}",
    config={
        "dataset": args.dataset,
        "seed": seed,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "participation_fraction": participation_fraction,
        "partitioner_type": partitioner_type,
        "partition_alpha": partition_alpha,
        "loss_type": loss_type,
        "noise_ratio": noise_ratio,
    },
)

name = f"{wandb.run.group}_{wandb.run.name}"

if partitioner_type == "iid":
    partitioner = IidPartitioner(
        num_partitions=num_clients
    )
elif partitioner_type == "dirichlet":
    partitioner = DirichletPartitioner(
        num_partitions=num_clients,
        alpha=partition_alpha,
        partition_by=partition_by,
    )

set_seed(seed)

if args.dataset == "cifar10_noisy":
    test_loader, get_client_loader = load_dataset(partitioner, batch_size=batch_size, noise_ratio=noise_ratio)
else:
    test_loader, get_client_loader = load_dataset(partitioner, batch_size=batch_size)

global_model = SimpleCNN(num_classes).to(DEVICE)
local_models = [SimpleCNN(num_classes).to(DEVICE) for _ in range(num_clients)]

# wandb.watch(global_model, log="all")
# wandb.watch(local_models, log="all")

os.makedirs(f"save/{name}", exist_ok=True)

for round in tqdm(range(num_rounds)):
    num_participating_clients = max(1, int(participation_fraction * num_clients))
    participating_clients = random.sample(range(num_clients), num_participating_clients)

    round_models = []
    for client_idx in tqdm(participating_clients, leave=False):
        trainloader, valloader = get_client_loader(client_idx)
        model = SimpleCNN(num_classes).to(DEVICE)
        optimizer = optim.Adam(model.parameters())
        criterion = criterions[loss_type]
        model.load_state_dict(global_model.state_dict())
        model.train()
        train_loss = 0
        for _ in tqdm(range(local_epochs), leave=False):
            for batch in tqdm(trainloader, leave=False):
                images, labels = process_batch(batch)
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                if loss_type == "ce":
                    loss = criterion(outputs, labels)
                    loss.backward()
                    train_loss += loss.item()
                elif loss_type == "dmi":
                    loss = criterion(outputs, labels, num_classes)
                    loss.backward()
                    train_loss += loss.item()
                else:                
                    loss, joints, marginals = criterion(outputs, labels)
                    (-loss).backward()
                    train_loss += (-loss).item()
                optimizer.step()
        round_models.append(model)

        test_loss, accuracy = evaluate(model, valloader, DEVICE, process_batch)
        local_models[client_idx].load_state_dict(model.state_dict())
        wandb.log(
            {
                str(client_idx): {
                    "local_train_loss": train_loss,
                    "local_test_loss": test_loss,
                    "local_accuracy": accuracy,
                }
            },
            commit=False,
        )
        torch.save(model.state_dict(), f"save/{name}/c_{round}_{client_idx}.pt")

    federated_averaging(global_model, round_models, DEVICE)
    test_loss, accuracy = evaluate(global_model, test_loader, DEVICE, process_batch)
    wandb.log(
        {
            "global_loss": test_loss,
            "global_accuracy": accuracy,
            "pariticipating_clients": participating_clients,
        }
    )
    torch.save(global_model.state_dict(), f"save/{name}/g_{round}.pt")
