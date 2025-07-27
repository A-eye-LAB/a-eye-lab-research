# Torch Imports
import torch
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import random_split

# Model Imports
import models
import torchvision.models as torchvision_models

# Module Imports
from modules.trainer import Trainer
from modules.datasets import MultiFolderDataset
from modules.losses import get_loss
from modules.optimizers import get_optimizer
from modules.schedulers import get_scheduler
from modules.utils import fix_seeds, setup_cudnn
from modules.utils import dir_set, load_yaml

# ETC Imports
import argparse
import wandb

fix_seeds(42)
setup_cudnn()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Configuration file to use")

    args = parser.parse_args()

    return args


def load_model(model_name, num_classes, pretrained):
    if hasattr(models, model_name):
        model = getattr(models, model_name)(num_classes=num_classes, pretrained=pretrained)
    else:
        model = getattr(torchvision_models, model_name)(num_classes=num_classes, pretrained=pretrained)
    return model

def main(cfg):
    device = torch.device(cfg['DEVICE'])

    wandb.init(
        project=cfg["WANDB_PROJECT"],
        job_type="train",
        config=cfg
    )

    dataset = MultiFolderDataset(cfg["DATASET"]["TRAIN_DATA_DIR"])

    # 전체 길이에서 80% 학습, 20% 검증 분할
    total_size = len(dataset)
    train_size = int(cfg["DATASET"]["RATIO"] * total_size)
    val_size = total_size - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["TRAIN"]["BATCH_SIZE"],
        num_workers=cfg["DATASET"]["NUM_WORKERS"],
        shuffle=True,
        pin_memory=True
    )

    valid_loader = DataLoader(
        val_set,
        batch_size=cfg["TRAIN"]["BATCH_SIZE"],
        num_workers=cfg["DATASET"]["NUM_WORKERS"],
        shuffle=False,

        pin_memory=True,
    )

    model = load_model(
        cfg["MODEL"]["NAME"], 
        cfg["MODEL"]["NUM_CLASSES"], 
        cfg["MODEL"]["PRETRAINED"]
    )
    model = model.to(device)

    # Loss Function & Optimizer Set
    criterion = get_loss(cfg["LOSS"]["NAME"])
    optimizer = get_optimizer(model, cfg["OPTIMIZER"])
    scheduler = get_scheduler(optimizer, **cfg["SCHEDULER"])
    scaler = GradScaler()

    save_dir, name = dir_set(cfg["SAVE_DIR"], model)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=cfg,
        device=device,
        wandb=wandb,
        checkpoint_dir=str(save_dir),
        train_loader=train_loader,
        valid_loader=valid_loader,
    )

    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_yaml(args.cfg)

    main(cfg)
