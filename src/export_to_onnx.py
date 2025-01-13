import os
import torch
from models.densenet import DenseNetTest

def get_latest_checkpoint(weights_dir):
    """
    Get the latest checkpoint file based on naming convention.
    
    Args:
        weights_dir (str): Path to the weights directory.
    
    Returns:
        str: Path to the latest checkpoint file.
    """
    checkpoints = sorted(
        [f for f in os.listdir(weights_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pt")],
        key=lambda x: int(x.split("_epoch_")[1].split(".")[0])
    )
    if not checkpoints:
        raise FileNotFoundError("No checkpoint files found in the directory.")
    return os.path.join(weights_dir, checkpoints[-1])

def export_to_onnx(weights_dir, output_onnx_path, num_classes=2, input_size=(1, 3, 224, 224), device="cpu"):
    """
    Export the model from the latest checkpoint to ONNX format.

    Args:
        weights_dir (str): Path to the directory containing weights.
        output_onnx_path (str): Path to save the ONNX model.
        num_classes (int): Number of output classes for the model.
        input_size (tuple): Input size for the model (batch_size, channels, height, width).
        device (str): Device to load the model (e.g., 'cpu', 'cuda').
    """
    # Find the latest checkpoint
    checkpoint_path = get_latest_checkpoint(weights_dir)
    print(f"Using checkpoint: {checkpoint_path}")

    # Load the model
    model = DenseNetTest(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()


    dummy_input = torch.randn(*input_size).to(device)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},  # Dynamic batch size
            "output": {0: "batch_size"}
        },
        opset_version=11
    )
    print(f"Model successfully exported to {output_onnx_path}")


if __name__ == "__main__":
    weights_dir = "/home/mia/a-eye-lab-research/outputs/DenseNetTest_20241201_201817/weights"
    output_onnx = "densenet_latest.onnx"
    export_to_onnx(weights_dir, output_onnx)
