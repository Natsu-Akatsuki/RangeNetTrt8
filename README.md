# RangeNetTrt8

## Purpose

1）**更新的依赖和 API**：将 [RangeNet 仓库](https://github.com/PRBonn/rangenet_lib)部署到 TensorRT8+，Ubuntu20.04+ 的环境中；移除 Boost 库；使用智能指针管理 TensorRT 对象和 GPU 显存的内存回收；提供 ROS 例程

2）**更快的运行速度**：修正了使用 FP16，分割精度降低的问题 [issue#9](https://github.com/PRBonn/rangenet_lib/issues/9)，使模型在保有精度的同时，预测速度大大提升；使用 CUDA 编程对数据进行预处理；使用 libtorch 对数据进行 KNN 后处理（参考 [Here](https://github.com/PRBonn/lidar-bonnetal/blob/master/train/tasks/semantic/postproc/KNN.py)）

<p align="center">
	<img src="assets/000000.png" alt="img" width=50% height=50% />
</p>

## Prerequisites

1）步骤 1：下载和解压缩 libtorch

> **Note**
>
> 使用过 Conda 环境的 Torch 库，然后发现其速度会相对较慢，后处理部分从 6 ms 到 30 ms

```bash
$ wget -c https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.10.2%2Bcu113.zip -O libtorch.zip
$ unzip libtorch.zip
```

2）步骤 2：搭建深度学习环境，可参考 [Here](https://natsu-akatsuki.github.io/ambook/#/Deep%20Learning/Setup)，已测试版本如下，至少需要用到 3000 M 的显存

| Ubuntu | TensorRT |      CUDA       |    cuDNN    |         —          |
|:------:|:--------:|:---------------:|:-----------:|:------------------:|
| 20.04  |  8.2.3   | CUDA 11.4.r11.4 | cuDNN 8.2.4 | :heavy_check_mark: |
| 22.04  | 8.2.5.1  | CUDA 11.3.r11.3 | cuDNN 8.8.0 | :heavy_check_mark: |
| 22.04  | 8.4.1.5  | CUDA 11.3.r11.3 | cuDNN 8.8.0 |        :x:         |

添加环境变量到 ~/.bashrc

```bash
# 示例配置：

# >>> 深度学习配置 >>>
# 导入CUDA环境
CUDA_PATH=/usr/local/cuda/bin
CUDA_LIB_PATH=/usr/local/cuda/lib64

# 导入TensorRT环境
export TENSORRT_DIR=${HOME}/Application/TensorRT-8.4.1.5/
TENSORRT_PATH=${TENSORRT_DIR}/bin
TENSORRT_LIB_PATH=${TENSORRT_DIR}/lib

# 导入libtorch环境
export Torch_DIR=${HOME}/Application/libtorch/share/cmake/Torch"

export PATH=${PATH}:${CUDA_PATH}:${TENSORRT_PATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_LIB_PATH}:${TENSORRT_LIB_PATH}
```

3）步骤 3：（可选，若需要使用 ROS 的相关组件）ROS1（Noetic），ROS2（Humble）

4）步骤 4：安装 apt 和 Python 包

```bash
$ sudo apt install build-essential python3-dev python3-pip apt-utils git cmake libboost-all-dev libyaml-cpp-dev libopencv-dev python3-empy
$ pip install catkin_tools trollius numpy
```

## Install

1）步骤一：导入仓库

```bash
$ git clone https://github.com/Natsu-Akatsuki/RangeNetTrt8 ~/rangetnet_pp/src
```

2）步骤二：导入模型文件（在 rangenet_pp/src 下解压缩 model.tar.gz 和新建 data 文件夹，并在该文件夹下按需下载实例代码），相关文件见[百度云](https://pan.baidu.com/s/1iXSWaEfZsfpRps1yvqMOrA?pwd=9394)

<details>
    <summary>目录结构</summary>

```bash
.
├── model
│   ├── arch_cfg.yaml
│   ├── data_cfg.yaml
│   └── model.onnx
├── data
└── ├── 000000.pcd
    ├── kitti_2011_09_30_drive_0027_synced
    └── kitti_2011_09_30_drive_0027_synced.bag
```

</details>

## Usage

首次运行需要等待一段时间，生成 TensorRT 优化引擎（engine）

<details>
    <summary>:wrench: <b>用例 1：</b>
        在 ROS1 或 ROS2 架构下跑数据
    </summary>
<p align="center">
	<img src="assets/ros.gif" alt="img" width=50% height=50% />
</p>



```bash
$ cd ~/rangetnet_pp/
$ catkin build
$ source devel/setup.bash
$ roslaunch rangenet_pp rangenet.launch
$ roslaunch rangenet_pp rosbag.launch
```

</details>

<details>
    <summary>:wrench: <b>用例 2：</b>
        预测单帧点云（PCD 格式）
    </summary>

> **Note**
>
> PCD 点云字段为 xyzi，强度字段（intensity）需要归一化（0-1）

```
# 修改 config/infer.yaml 中的配置参数
$ mkdir build
$ cd build
$ cmake .. && make
$ ./demo
```

</details>

<details>
    <summary>:wrench: <b>用例 3：</b>
        显示预处理、预测时间、后处理时间
    </summary>

```bash
# 对于用例 1
（ROS1）$ catkin build --cmake-args -DPERFORMANCE_LOG=ON
（ROS2）$ colcon build --cmake-args -DPERFORMANCE_LOG=ON

# 对于用例 2
$ cmake -DPERFORMANCE_LOG=ON .. 
```

</details>


 

## FAQ

<details>
    <summary>:question: <b>问题 1：</b>
        [libprotobuf ERROR google/protobuf/text_format.cc:298] Error parsing text-format onnx2trt_onnx.ModelProto: 1:1:
    </summary>

1）情况一：下载的 ONNX 模型不完整，模型解析出问题。重新下载即可。

</details>

## Roadmap

- [ ] 追加 Pybind11 实现
- [ ] 追加英文文档
- [ ] 解决算法随机性的问题
- [ ] 提供 Docker 环境
- [ ] 解决 [issue#8](https://github.com/Natsu-Akatsuki/RangeNetTrt8/issues/8)
