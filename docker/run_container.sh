docker run -it \
-v /home/inha/ros_vision:/workspace \
-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
-e DISPLAY=unix$DISPLAY \
--net=host \
--gpus all \
--privileged \
--name vision_trt \
nvcr.io/nvidia/tensorrt:22.12-py3
