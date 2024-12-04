# Docker installation

Docker can ensure that all developers in a project have a common, consistent development environment. It is recommended for beginners, casual users, people who are unfamiliar with Ubuntu.

## Step 1: Installing dependencies manually

- Install Nvidia driver
- Install Docker Engine
- Install Docker compose
- Install NVIDIA Container Toolkit

## Step 2: Import project

```bash
$ git clone https://github.com/Natsu-Akatsuki/RangeNet-TensorRT ~/rangenet/src/rangenet
$ wget -c https://github.com/Natsu-Akatsuki/RangeNet-TensorRT/releases/download/v0.0.0-alpha/model.onnx -O ~/rangenet/src/rangenet/model/model.onnx
```

## Step 2 (Alternative): Pulling the image from Docker Hub

See more in https://hub.docker.com/repository/docker/877381/rangenet/general.

| Version                                                    | Status                                                       |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| 877381/rangenet:ubuntu20.04_cuda11.1.1_tensorrt10.6_noetic | [![Build image ubuntu20.04_cuda11.1.1_tensorrt10.6_noetic](https://github.com/Natsu-Akatsuki/RangeNet-TensorRT/actions/workflows/build_image_ubuntu20.04_cuda11.1.1_tensorrt10.6_noetic.yml/badge.svg)](https://github.com/Natsu-Akatsuki/RangeNet-TensorRT/actions/workflows/build_image_ubuntu20.04_cuda11.1.1_tensorrt10.6_noetic.yml) |
| 877381/rangenet:ubuntu20.04_cuda12.4.1_tensorrt10.6_noetic | ![Build image ubuntu20.04_cuda12.4.1_tensorrt10.6_noetic](https://github.com/Natsu-Akatsuki/RangeNet-TensorRT/actions/workflows/build_image_ubuntu20.04_cuda12.4.1_tensorrt10.6_noetic.yml/badge.svg) |
| 877381/rangenet:ubuntu22.04_cuda12.4.1_tensorrt10.6_humble | [![Build image ubuntu22.04_cuda12.4.1_tensorrt10.6_humble](https://github.com/Natsu-Akatsuki/RangeNet-TensorRT/actions/workflows/build_image_ubuntu22.04_cuda12.4.1_tensorrt10.6_humble.yml/badge.svg)](https://github.com/Natsu-Akatsuki/RangeNet-TensorRT/actions/workflows/build_image_ubuntu22.04_cuda12.4.1_tensorrt10.6_humble.yml) |

The image size is about 20-26 GB (after uncompressed), so make sure you have enough space

```bash
# e.g. docker pull 877381/rangenet:ubuntu20.04_cuda12.4.1_tensorrt10.6_noetic
$ docker pull <image_name>

# Rename the image (remove the Docker user prefix)
$ TAG=ubuntu20.04_cuda12.4.1_tensorrt10.6_noetic && docker tag 877381/rangenet:${TAG} rangenet:${TAG}
```

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
# {22.04, 20.04}
ubuntu_version = "20.04"
# {noetic, humble}
ros_version = "noetic"
tensorrt_version = "10.6"
# {11.1.1, 12.4.1 (require: nvidia-driver 550)}
cuda_version = "12.4.1"
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

## Step 3: Run the container

After the image is built, you can run the container with the following command

```bash
# Please replace the image name with the name of the image you built or pulled
$ cd ~/rangenet/src/rangenet/docker/ubuntu20.04_cuda11.1.1_tensorrt10.6_noetic

# Please change the mount path in docker-compose.yml
$ docker compose up
# Open an another terminal
$ docker exec -it rangenet /bin/bash
```

## Step 4: Quick start

```bash
# In container
$ cd ~/rangenet/src/rangenet/
$ mkdir build
$ unset ROS_VERSION && cd build && cmake .. && make -j4
$ ./demo 
```

## Reference

- https://github.com/pytorch/TensorRT/blob/main/docker/Dockerfile
- https://github.com/Natsu-Akatsuki/RangeNet-TensorRT/blob/422bfb0f97f91e7363de8b9fff3131fdc1558547/docker/Dockerfile-tensorrt8.2.2

