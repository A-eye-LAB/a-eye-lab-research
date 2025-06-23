from module.dataset import CustomImageDataset
from module.predict import onnx_predict, pytorch_predict
from module.evaluate import evaluate
from torch.utils.data import DataLoader

def main(dataset_path, model_path):

    dataset = CustomImageDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=16)

    if model_path.endswith('.pt'):
        y_true, y_pred = pytorch_predict(model_path, dataloader)
    elif model_path.endswith('.onnx'):
        y_true, y_pred = onnx_predict(model_path, dataloader)
    else:
        raise ValueError("지원하지 않는 모델 형식입니다. .pt 또는 .onnx 파일을 사용해주세요.")

    metrics = evaluate(y_true, y_pred)

    for metric, value in metrics.items():
        if metric == "Confusion Matrix":
            print(f"\n{metric}")
            print(value, "\n")
        else:
            print(f"{metric:<15} : {value:.4f}")

if __name__ == "__main__":
    main(
        dataset_path="/workspace/a-eye-lab-research/real_data",
        model_path="/workspace/model_fastvit.onnx"
    )