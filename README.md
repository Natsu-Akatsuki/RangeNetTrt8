# [RangeNetTrt8](https://github.com/Natsu-Akatsuki/RangeNetTrt8)

本环境配置面向[rangenet](https://github.com/StephenYang190/rangenet_lib)项目旨将原工程部署到TensorRT8，ubuntu20.04中，:) 迁移完才发现[有人](https://github.com/StephenYang190/rangenet_lib)已经做过一样的工作

## 方法一：docker

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

## 方法二：native PC

### 依赖

- ros1
- nvidia driver

- TensorRT 8.0.03（tar包下载）, cuda_11.2.r11.2 cudnn 8.1.1（理论上使用其他版本的Trt也行，做好cuda等版本的适配即可）

- apt package and python package

```bash
$ sudo apt install build-essential python3-dev python3-pip apt-utils git cmake libboost-all-dev libyaml-cpp-dev libopencv-dev python3-empy
$ pip install catkin_tools trollius numpy
```

- 创建工作空间

```bash
$ git clone https://github.com/Natsu-Akatsuki/RangeNetTrt8 ~/RangeNetTrt8/src
```

- 下载onnx模型

```bash
$ wget -c http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/predictions/darknet53.tar.gz -O ~/RangeNetTrt8/src/darknet53.tar.gz
$ cd ~/RangeNetTrt8/src && tar -xzvf darknet53.tar.gz
```

### 安装

- 修改CMakeLists：将CMakeLists_v2.txt替换为CMakeLists.txt，修改其中的TensorRT等依赖库的路径
- 编译

```bash
$ cd ~/RanageNetTrt8/src
$ bash script/build_container_rangenet.sh
# 编译和执行
$ cd ~/RanageNetTrt8
$ catkin_build
```

- 执行

```bash
$ ~/RanageNetTrt8/devel/lib/rangenet_lib/infer -s ~/RanageNetTrt8/src/example/000000.bin -p ~/RanageNetTrt8/src/darknet53 -v
# s: sample
# p: model dir
# v: output verbose log
```

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
