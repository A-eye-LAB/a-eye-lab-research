docker run --gpus all -itd \
  --ipc=host \
  -v /home/hoon/a-eye-lab-research:/workspace/a-eye-lab-research \
  -v /home/hoon/cataract_data:/workspace/data \
  --name onnx_container \
  onnx-gpu