# [RangeNetTrt8](https://github.com/Natsu-Akatsuki/RangeNetTrt8)

本工程旨将[rangenet工程](https://github.com/PRBonn/rangenet_lib)部署到TensorRT8，ubuntu20.04中

## **Attention**

- **最近正在进行大幅度地改动，本仓库暂时不能很稳定地使用~**
- 由于使用了较新的API，本工程只适用于TensorRT8.2.3，但可自行查文档修改相应的API
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

## TODO

- [ ] 去掉0均值1方差的数据预处理，重新训练模型（毕竟已经有BN层了）
- [ ] fix: 每次运行的结果不一样...（就很迷）

## 文件树

├── **build**    
├── **devel**   
├── **logs**   
└── **src**   
　└── **RangeNetTrt8**  
　　├── **CMakeLists.txt**   
　　├── **CMakeLists_v2.txt**   
　　├── **darknet53**   
　　├── **docker**   
　　├── **example**   
　　├── **include**   
　　├── **launch**   
　　├── **LICENSE**   
　　├── **ops**   
　　├── **package.xml**   
　　├── **pics**   
　　├── **README.md**   
　　├── **rosbag**   
　　├── **script**   
　　├── **src**   
　　└── **utils**  

## 方法一：docker（改进ing）

### 依赖

- nvidia driver

- [docker](https://ambook.readthedocs.io/zh/latest/docker/rst/docker-practice.html#docker)
- [nvidia-container2](https://ambook.readthedocs.io/zh/latest/docker/rst/docker-practice.html#id4)

- 创建工作空间

```bash
$ git clone https://github.com/Natsu-Akatsuki/RangeNetTrt8 ~/docker_ws/RangeNetTrt8/src
```

- 下载onnx模型

```bash
$ wget -c http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/predictions/darknet53.tar.gz -O ~/docker_ws/RangeNetTrt8/src/darknet53.tar.gz
$ cd ~/docker_ws/RangeNetTrt8/src && tar -xzvf darknet53.tar.gz
```

### 安装

- 拉取镜像（镜像大小约为20G，需预留足够的空间）

```bash
$ docker pull registry.cn-hangzhou.aliyuncs.com/gdut-iidcc/rangenet:1.0
```

- 创建容器

```bash
$ cd ~/docker_ws/RanageNetTrt8/src
$ bash script/build_container_rangenet.sh
# 编译和执行
(docker) $ cd /docker_ws/RanageNetTrt8
(docker) $ catkin_build
(docker) $ /docker_ws/RanageNetTrt8/devel/lib/rangenet_lib/infer -s /docker_ws/RanageNetTrt8/src/example/000000.bin -p /docker_ws/RanageNetTrt8/src/darknet53 -v
# s: sample
# p: model dir
# v: output verbose log
```

**NOTE**

首次运行生成TensorRT模型运行需要一段时间

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image.png" alt="img" style="zoom:80%;" />

## 方法二：native PC

### 依赖

- **ros1**
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
$ wget -c http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/predictions/darknet53.tar.gz -O ~/RangeNetTrt8/src/darknet53.tar.gz
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
CUDA_LIB_PATH=/usr/local/cuda/lib64
TENSORRT_LIB_PATH=${HOME}/application/TensorRT-8.2.3.0/lib
PYTORCH_LIB_PATH=${HOME}/application/libtorch/lib
export PATH=${PATH}:${CUDA_PATH}:"~/bin"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_LIB_PATH}:${TENSORRT_LIB_PATH}:${PYTORCH_LIB_PATH}
```

### 安装

- 修改CMakeLists：修改其中的TensorRT, libtorch等依赖库的路径
- 编译和执行

```bash
# 编译和执行
$ cd ~/RanageNetTrt8
$ catkin_build
$ source devel/setup.bash
$ roslaunch rangenet_plusplus rangenet.launch
# 播放包（该模型仅适用于kitti数据集，需自行下载包文件和修改该launch文档）
$ roslaunch rangenet_plusplus rosbag.launch
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/ros.gif" alt="img" style="zoom:67%;" />

**NOTE**

首次运行生成TensorRT模型运行需要一段时间

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