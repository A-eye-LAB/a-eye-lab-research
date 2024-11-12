docker build -t cataract:v0.1 .

docker run --gpus all -itd \
  --ipc=host \
  -v /home/hoon/cataract:/workspace/data \
  --name cataract_container \
  cataract:v0.1