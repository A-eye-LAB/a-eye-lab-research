from ultralytics import YOLO

# Load a model
model = YOLO("yolo11l-cls.pt") 


# Train the model with a custom YAML configuration
results = model.train(data="/workspace/data/catract",  # Update to a single YAML file path
                      epochs=100, 
                      imgsz=224,
                      project="/workspace/a-eye-lab-research/yolo/output",
                      name="cataract_yolo11l-combined_data",
                      exist_ok=True)


