# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler

# Model Imports
from models.mlp import MLP_TEST

# Module Imports
from modules.trainer import Trainer
from modules.datasets import get_mnist_dataloaders
from modules.losses import get_loss
from modules.optimizers import get_optimizer
from modules.schedulers import get_scheduler
from modules.utils import fix_seeds, setup_cudnn
from modules.utils import dir_set, load_yaml

# ETC Imports
import argparse

fix_seeds(42)
setup_cudnn()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', 
                        type=str, 
                        required=True, 
                        help='Configuration file to use'
                        )

    parser.add_argument('--resume',
                        action='store_true',
                        help="Resume training",
                        )

    args = parser.parse_args()

    return args

def main(cfg, resume=False):
    
    device = torch.device(cfg['DEVICE'])


    train_loader, valid_loader = get_mnist_dataloaders(
        batch_size=cfg["TRAIN"]["BATCH_SIZE"],
        num_workers=cfg["DATASET"]["NUM_WORKERS"]
    )

    # Model Set
    model = eval('{}(num_classes={}, pretrained={})'.format(
        cfg["MODEL"]["NAME"], 
        cfg["MODEL"]["NUM_CLASSES"],
        cfg["MODEL"]["PRETRAINED"]
        ))
    model = model.to(device)

    # Loss Function & Optimizer Set
    criterion = get_loss(cfg['LOSS']['NAME'])
    optimizer = get_optimizer(model, cfg['OPTIMIZER'])
    scheduler = get_scheduler(optimizer, **cfg['SCHEDULER'])
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
        checkpoint_dir=save_dir,
        train_loader=train_loader,
        valid_loader=valid_loader,
        resume=resume,
    )

    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    cfg = load_yaml(args.cfg)

    main(cfg, args.resume)
