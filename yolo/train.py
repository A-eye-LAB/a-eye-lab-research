from ultralytics import YOLO
import torch
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='data.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--workers', type=int, default=16)
    return parser.parse_args()

def train_yolo(cfg):
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Try to load the model
    model = YOLO('yolov8m.pt')

    args = {
        'data': cfg.config,          # path to data config file
        'epochs': cfg.epochs,                   # number of epochs
        'imgsz': 640,                    # image size
        'batch': cfg.batch,                     # batch size
        'patience': cfg.patience,                  # early stopping patience
        'device': device,                # cuda device, i.e. 0 or 0,1,2,3 or cpu
        'workers': cfg.workers,                    # number of worker threads
        'project': 'runs',         # save to project/name
        'name': 'image',                   # save to project/name
        'exist_ok': False,               # existing project/name ok, do not increment
        'pretrained': True,              # use pretrained model
        'optimizer': 'auto',             # optimizer (SGD, Adam, etc.)
        'verbose': True,                 # print verbose output
        'augment':True,
        'seed': 0,                       # random seed for reproducibility
        'deterministic': True,           # deterministic training
    }

    print("Starting training...")
    # Start training
    results = model.train(**args)

    # Save the trained model
    model.save('best.pt')
    print("Training completed! Model saved as 'best.pt'")

if __name__ == '__main__':
    args = args_parser()
    train_yolo(args)