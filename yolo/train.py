from ultralytics import YOLO
import yaml
import os
import torch


def train_yolo():
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Try to load the model
    model = YOLO('yolo11s.pt')

    # Load dataset configuration
    with open('dataset.yaml', 'r') as f:
        data_config = yaml.safe_load(f)

    # Training arguments
    args = {
        'data': 'dataset.yaml',          # path to data config file
        'epochs': 500,                   # number of epochs
        'imgsz': 640,                    # image size
        'batch': 16,                     # batch size
        'patience': 50,                  # early stopping patience
        'device': device,                # cuda device, i.e. 0 or 0,1,2,3 or cpu
        'workers': 8,                    # number of worker threads
        'project': 'runs/train',         # save to project/name
        'name': 'image_2310_yolov11s',                   # save to project/name
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
    train_yolo()
