FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

RUN sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list \
    && rm /etc/apt/sources.list.d/* \
    && apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y \
        apt-utils \
        bash-completion \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        gcc-9 g++-9 gcc-10 g++-10 \
        git \
        keyboard-configuration \
        libboost-all-dev \
        libfmt-dev \
        libx11-dev \
        libyaml-cpp-dev \
        locales \
        lsb-core \
        mlocate \
        nano \
        net-tools \
        openssh-server \
        python3-dev \
        python3-empy \
        python3-pip \
        software-properties-common \
        sudo \
        unzip \
        vim \
        wget \
    && rm -rf /var/lib/apt/lists/*


RUN DEBIAN_FRONTEND=noninteractive sh -c '. /etc/lsb-release && echo "deb http://mirrors.tuna.tsinghua.edu.cn/ros/ubuntu/ `lsb_release -cs` main" > /etc/apt/sources.list.d/ros-latest.list' \
    && apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
    && apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y ros-noetic-desktop-full \
        python3-catkin-tools \
        python3-rosdep \
        python3-rosinstall \
        python3-rosinstall-generator \
        python3-wstool \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# >>> nvidia-container-runtime >>>
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV __NV_PRIME_RENDER_OFFLOAD=1 
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia


# >>> Install TensorRT >>>
ENV TENSORRT_VERSION 10.6
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub \
    && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" \
    && apt update \
    && TENSORRT_MAJOR_VERSION=`echo ${TENSORRT_VERSION} | cut -d '.' -f 1` \
    && DEBIAN_FRONTEND=noninteractive apt install -y \
        libnvinfer${TENSORRT_MAJOR_VERSION}=${TENSORRT_VERSION}.* \
        libnvinfer-plugin${TENSORRT_MAJOR_VERSION}=${TENSORRT_VERSION}.* \
        libnvinfer-dev=${TENSORRT_VERSION}.* \
        libnvinfer-headers-dev=${TENSORRT_VERSION}.* \
        libnvinfer-headers-plugin-dev=${TENSORRT_VERSION}.* \
        libnvinfer-plugin-dev=${TENSORRT_VERSION}.* \
        libnvonnxparsers${TENSORRT_MAJOR_VERSION}=${TENSORRT_VERSION}.* \
        libnvonnxparsers-dev=${TENSORRT_VERSION}.* \
    && rm -rf /var/lib/apt/lists/*


SHELL ["/bin/bash", "-c"]

# Fix https://github.com/ros-visualization/rviz/issues/1780.
RUN DEBIAN_FRONTEND=noninteractive add-apt-repository ppa:beineri/opt-qt-5.12.10-focal -y \
    && apt update \
    && apt install -y qt512charts-no-lgpl qt512svg qt512xmlpatterns qt512tools qt512translations qt512graphicaleffects qt512quickcontrols2 qt512wayland qt512websockets qt512serialbus qt512serialport qt512location qt512imageformats qt512script qt512scxml qt512gamepad qt5123d
    
# Install hstr
RUN DEBIAN_FRONTEND=noninteractive add-apt-repository ppa:ultradvorka/ppa -y \
    && apt update \
    && apt update && apt install -y hstr \
    && rm -rf /var/lib/apt/lists/*


ENTRYPOINT ["/bin/bash"]

# >>> Change user permissions >>>
ARG USER_NAME=rangenet
# Set the password '123' for the 'rangenet' user
RUN useradd ${USER_NAME} -m -G sudo -u 1000 -s /bin/bash && echo ${USER_NAME}:123 | chpasswd
USER ${USER_NAME}

WORKDIR /home/${USER_NAME}
RUN echo "source /opt/qt512/bin/qt512-env.sh" >> ~/.bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# >>> Download libtorch >>>
RUN cd ~ \
    && wget -c https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcu111.zip \
    && unzip libtorch-cxx11-abi-shared-with-deps-1.10.0+cu111.zip \
    && rm libtorch-cxx11-abi-shared-with-deps-1.10.0+cu111.zip \
    && echo "export Torch_DIR=${HOME}/libtorch/share/cmake/Torch" >> ~/.bashrc
    
# >>> Download CMake >>>
RUN pip3 install --user cmake==3.18 \
    && echo 'export PATH=${HOME}/.local/bin:${PATH}' >> ~/.bashrc

