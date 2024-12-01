FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

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


RUN curl --resolve raw.githubusercontent.com:443:185.199.108.133 -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y ros-humble-desktop python3-colcon-common-extensions ros-humble-pcl-ros


# >>> nvidia-container-runtime >>>
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV __NV_PRIME_RENDER_OFFLOAD=1 
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia


# >>> Install TensorRT >>>
ENV TENSORRT_VERSION 10.6
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
    && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
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

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# >>> Download libtorch >>>
RUN cd ~ \
    && wget -c https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcu111.zip \
    && unzip libtorch-cxx11-abi-shared-with-deps-1.10.0+cu111.zip \
    && rm libtorch-cxx11-abi-shared-with-deps-1.10.0+cu111.zip \
    && echo "export Torch_DIR=${HOME}/libtorch/share/cmake/Torch" >> ~/.bashrc
    
# Fix https://github.com/PointCloudLibrary/pcl/pull/5252
RUN cd ~ \
    && wget -c https://github.com/PointCloudLibrary/pcl/releases/download/pcl-1.13.1/source.zip \
    && unzip source.zip \
    && rm source.zip \
    && cd pcl \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_visualization=ON -DBUILD_apps=OFF -DBUILD_examples=OFF -DBUILD_tools=OFF -DBUILD_samples=OFF .. \
    && make -j4 \
    && echo "123" | sudo -S make install \
    && cd ~ \
    && rm -rf pcl

