# [RangeNetTrt8](https://github.com/Natsu-Akatsuki/RangeNetTrt8)

本工程旨将[rangenet工程](https://github.com/PRBonn/rangenet_lib)部署到TensorRT8，ubuntu20.04中

![image-20220330012729619](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220330012729619.png)

## **Attention**

- 由于使用了较新的API，本工程**只适用于TensorRT8.2.3+**，但可自行查文档修改相应的API
- 使用过conda环境的torch，然后发现速度会相对较慢(6ms->30ms)

## Feature

更新的依赖和API

- 将代码部署环境提升到**TensorRT8**, **ubuntu20.04**
- 提供**docker**环境
- 移除**boost**库
- 使用**智能指针**管理tensorrt对象和GPU显存的内存回收
- 提供**ros**例程

更快的运行速度

- 修正了使用**FP16**，分割精度降低的问题[issue#9](https://github.com/PRBonn/rangenet_lib/issues/9)。使模型在保有精度的同时，预测速度大大提升
- 使用**cuda**编程对数据进行预处理
- 使用**libtorch**对数据进行knn后处理([参考代码here](https://github.com/PRBonn/lidar-bonnetal/blob/master/train/tasks/semantic/postproc/KNN.py))

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220227223539620.png" alt="image-20220227223539620" style="zoom:80%;" />

## File Tree

```
.
├── cmake
│   └── TensorRT.cmake
├── CMakeLists.txt
├── config
│   └── infer.yaml
├── darknet53
│   ├── arch_cfg.yaml
│   ├── backbone
│   ├── data_cfg.yaml
│   ├── model.onnx
│   ├── model.trt
│   ├── segmentation_decoder
│   └── segmentation_head
├── data
│   ├── 000000.pcd
│   └── 002979.pcd
├── docker
├── include
│   ├── net.hpp
│   └── netTensorRT.hpp
├── launch
│   ├── rangenet.launch
│   ├── rosbag.launch
│   └── rviz.rviz
├── LICENSE
├── package.xml
├── README.md
└── src
    ├── network
    ├── ops
    ├── semantic_segmentation_node.cpp
    ├── single_shot_demo.cpp
    └── utils
```

## Usage

### Requirement

- **ros1 noetic**
- **nvidia driver**

- **TensorRT 8.2.3**（tar包下载）, **cuda_11.4.r11.4**,  **cudnn 8.2.4**

- apt package and python package

```bash
$ sudo apt install build-essential python3-dev python3-pip apt-utils git cmake libboost-all-dev libyaml-cpp-dev libopencv-dev python3-empy
$ pip install catkin_tools trollius numpy
```

- 创建工作空间

```bash
$ git clone https://github.com/Natsu-Akatsuki/RangeNetTrt8 ~/RangeNetTrt8/src
```

- 下载**onnx**模型

```bash
$ wget -c https://www.ipb.uni-bonn.de/html/projects/semantic_suma/darknet53.tar.gz -O ~/RangeNetTrt8/src/darknet53.tar.gz
$ cd ~/RangeNetTrt8/src && tar -xzvf darknet53.tar.gz
```

- 下载**libtorch**

```bash
$ wget -c https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.10.2%2Bcu113.zip -O libtorch.zip
$ unzip libtorch.zip
```

TIP: 需导入各种环境变量到`~/.bashrc`

```bash
# example
export PATH="/home/helios/.local/bin:$PATH"
CUDA_PATH=/usr/local/cuda/bin
TENSORRT_PATH=${HOME}/application/TensorRT-8.2.3.0/bin
CUDA_LIB_PATH=/usr/local/cuda/lib64
TENSORRT_LIB_PATH=${HOME}/application/TensorRT-8.2.3.0/lib
PYTORCH_LIB_PATH=${HOME}/application/libtorch/lib
export PATH=${PATH}:${CUDA_PATH}:${TENSORRT_PATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_LIB_PATH}:${TENSORRT_LIB_PATH}:${PYTORCH_LIB_PATH}
```

### Install and Build

- 修改`CMakeLists`：修改其中的`TensorRT`, `libtorch`等依赖库的路径
- 编译和执行

```bash
# 编译和执行
$ cd ~/RanageNetTrt8
$ catkin_build
$ source devel/setup.bash

# dem01:
$ roslaunch rangenet_plusplus rangenet.launch
# 播放包（该模型仅适用于kitti数据集，需自行下载包文件和修改该launch文档）
$ roslaunch rangenet_plusplus rosbag.launch

# demo2:
# 需修改config/infer.yaml中的配置参数
$ ./devel/lib/rangenet_plusplus/single_shot_demo
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/ros.gif" alt="img" style="zoom:67%;" />

**NOTE**

首次运行生成TensorRT模型运行需要一段时间

## TODO

- [ ] 去掉0均值1方差的数据预处理，重新训练模型（毕竟已经有BN层了）
- [ ] fix: 每次运行的结果不一样...（就很迷）

## Q&A

- 模型解析出问题（查看是否下载的onnx模型是否完整，是否在解压缩时broken了）

> [libprotobuf ERROR google/protobuf/text_format.cc:298] Error parsing text-format onnx2trt_onnx.ModelProto: 1:1: Invalid control characters encountered in text. 
> [libprotobuf ERROR google/protobuf/text_format.cc:298] Error parsing text-format onnx2trt_onnx.ModelProto: 1:14: Message type "onnx2trt_onnx.ModelProto" has no field named "pytorch". Message type "onnx2trt_onnx.ModelProto" has no field named "pytorch"

## Citations

If you use this library for any academic work, please cite the original [paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf).

```
@inproceedings{milioto2019iros,
  author    = {A. Milioto and I. Vizzo and J. Behley and C. Stachniss},
  title     = {{RangeNet++: Fast and Accurate LiDAR Semantic Segmentation}},
  booktitle = {IEEE/RSJ Intl.~Conf.~on Intelligent Robots and Systems (IROS)},
  year      = 2019,
  codeurl   = {https://github.com/PRBonn/lidar-bonnetal},
  videourl  = {https://youtu.be/wuokg7MFZyU},
}
```

If you use SuMa++, please cite the corresponding [paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/chen2019iros.pdf):

```
@inproceedings{chen2019iros, 
  author    = {X. Chen and A. Milioto and E. Palazzolo and P. Giguère and J. Behley and C. Stachniss},
  title     = {{SuMa++: Efficient LiDAR-based Semantic SLAM}},
  booktitle = {Proceedings of the IEEE/RSJ Int. Conf. on Intelligent Robots and Systems (IROS)},
  year      = {2019},
  codeurl   = {https://github.com/PRBonn/semantic_suma/},
  videourl  = {https://youtu.be/uo3ZuLuFAzk},
}
```

## License

Copyright 2019, Xieyuanli Chen, Andres Milioto, Jens Behley, Cyrill Stachniss, University of Bonn.

This project is free software made available under the MIT License. For details see the LICENSE file.