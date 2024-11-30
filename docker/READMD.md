# Docker installation

Docker can ensure that all developers in a project have a common, consistent development environment. It is recommended for beginners, casual users, people who are unfamiliar with Ubuntu.

## Step 1: Installing dependencies manually

- Install Nvidia driver
- Install Docker Engine
- Install Docker compose
- Install NVIDIA Container Toolkit

## Step 2: Create image by Dockerfile

- Open `generate_dockerfile.py` and modify the content to suit your needs.

```python
# {22.04, 20.04}
UBUNTU_VERSION = "20.04"
# {noetic, humble}
ROS_VERSION = "noetic"
# {10.6}
TENSORRT_VERSION = "10.6"
```

- If you want to use other version CUDA and cuDNN, you can modify the basic image in `generate_dockerfile.py`. See more images in https://hub.docker.com/r/nvidia/cuda/tags.

```python
def set_basic_image():
    tmp_dict = {
        "22.04": "nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04",
        "20.04": "nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04"}
    return tmp_dict[UBUNTU_VERSION]
```

- Run `generate_dockerfile.py` to generate the Dockerfile

```bash
$ python3 generate_dockerfile.py
```

- cd to the directory containing the Dockerfile and run the following command to build the image

```bash
# Here is an example of building the image
$ cd ubuntu20.04_cuda11.1.1_tensorrt10.6_noetic
$ bash build_image.sh
```

## Step 2 (Alternative): Pulling the image from Docker Hub

The image size is about 26.2G, so make sure you have enough space

```bash
$ docker pull
```

## Step 3: Run the container

After the image is built, you can run the container with the following command

```bash
$ docker compose up
# Open an another terminal
$ docker exec -it rangenet /bin/bash
```

## Step 4: Quick start

```bash
# In container
$ git clone https://github.com/Natsu-Akatsuki/RangeNet-TensorRT ~/workspace/rangenet/src
$ cd ~/workspace/rangenet/src/model
$ wget -c https://github.com/Natsu-Akatsuki/RangeNet-TensorRT/releases/download/v0.0.0-alpha/model.onnx
$ mkdir build
$ cd build && cmake .. && make -j4
$ ./demo 
```

## Reference

- https://github.com/pytorch/TensorRT/blob/main/docker/Dockerfile
- https://github.com/Natsu-Akatsuki/RangeNet-TensorRT/blob/422bfb0f97f91e7363de8b9fff3131fdc1558547/docker/Dockerfile-tensorrt8.2.2

