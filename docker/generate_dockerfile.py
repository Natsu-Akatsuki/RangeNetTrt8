"""
Refer to
1. https://github.com/pytorch/TensorRT/blob/main/docker/Dockerfile
"""

from pathlib import Path
import os

# {22.04, 20.04}
UBUNTU_VERSION = "20.04"
# {noetic, humble}
ROS_VERSION = "noetic"
TENSORRT_VERSION = "10.6"
# {11.1, 12.6.2}
CUDA_VERSION = "11.1.1"

target_dir = Path(__file__).resolve().parent
prefix = f"ubuntu{UBUNTU_VERSION}_cuda{CUDA_VERSION}_tensorrt{TENSORRT_VERSION}_{ROS_VERSION}"
target_dir = target_dir / prefix
target_dir.mkdir(parents=True, exist_ok=True)


def set_basic_image():
    tmp_dict = {
        "22.04": f"nvidia/cuda:{CUDA_VERSION}-cudnn-devel-ubuntu22.04",
        "20.04": f"nvidia/cuda:{CUDA_VERSION}-cudnn8-devel-ubuntu20.04"}
    return tmp_dict[UBUNTU_VERSION]


def set_ros():
    tmp_dict = {
        "noetic": """
RUN DEBIAN_FRONTEND=noninteractive sh -c '. /etc/lsb-release && echo "deb http://mirrors.tuna.tsinghua.edu.cn/ros/ubuntu/ `lsb_release -cs` main" > /etc/apt/sources.list.d/ros-latest.list' \\
    && apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \\
    && apt update \\
    && DEBIAN_FRONTEND=noninteractive apt install -y ros-noetic-desktop-full \\
        python3-catkin-tools \\
        python3-rosdep \\
        python3-rosinstall \\
        python3-rosinstall-generator \\
        python3-wstool \\
        python3-pip \\
    && rm -rf /var/lib/apt/lists/*""",
        "humble": """
RUN DEBIAN_FRONTEND=noninteractive curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \\
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \\
    && apt update
    && apt install ros-humble-desktop
"""
    }

    return tmp_dict[ROS_VERSION]


def set_apt():
    return """
RUN sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list \\
    && apt update \\
    && DEBIAN_FRONTEND=noninteractive apt install -y \\
        apt-utils \\
        bash-completion \\
        build-essential \\
        ca-certificates \\
        cmake \\
        curl \\
        gcc-9 g++-9 gcc-10 g++-10 \\
        git \\
        keyboard-configuration \\
        libboost-all-dev \\
        libfmt-dev \\
        libx11-dev \\
        libyaml-cpp-dev \\
        locales \\
        lsb-core \\
        mlocate \\
        nano \\
        net-tools \\
        openssh-server \\
        python3-dev \\
        python3-empy \\
        python3-pip \\
        software-properties-common \\
        vim \\
        wget \\
    && rm -rf /var/lib/apt/lists/*
"""


def set_nvidia():
    return """
# >>> nvidia-container-runtime >>>
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV __NV_PRIME_RENDER_OFFLOAD=1 
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia
"""


def set_tensort():
    tmp_str = """
# >>> Install TensorRT >>>
"""

    tmp_str += f"ENV TENSORRT_VERSION {TENSORRT_VERSION}\n"

    if UBUNTU_VERSION == "20.04":
        tmp_str += """RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub \\
    && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" \\
"""
    if UBUNTU_VERSION == "22.04":
        tmp_str += """RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \\
    && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \\
"""

    tmp_str += """    && apt update \\
    && TENSORRT_MAJOR_VERSION=`echo ${TENSORRT_VERSION} | cut -d '.' -f 1` \\
    && DEBIAN_FRONTEND=noninteractive apt install -y \\
        libnvinfer${TENSORRT_MAJOR_VERSION}=${TENSORRT_VERSION}.* \\
        libnvinfer-plugin${TENSORRT_MAJOR_VERSION}=${TENSORRT_VERSION}.* \\
        libnvinfer-dev=${TENSORRT_VERSION}.* \\
        libnvinfer-headers-dev=${TENSORRT_VERSION}.* \\
        libnvinfer-headers-plugin-dev=${TENSORRT_VERSION}.* \\
        libnvinfer-plugin-dev=${TENSORRT_VERSION}.* \\
        libnvonnxparsers${TENSORRT_MAJOR_VERSION}=${TENSORRT_VERSION}.* \\
        libnvonnxparsers-dev=${TENSORRT_VERSION}.* \\
    && rm -rf /var/lib/apt/lists/*
"""
    return tmp_str


def set_shell():
    return """
SHELL ["/bin/bash", "-c"]
"""


def set_others():
    tmp_str = ""
    if UBUNTU_VERSION == "20.04":
        tmp_str += """# Fix https://github.com/ros-visualization/rviz/issues/1780.
RUN DEBIAN_FRONTEND=noninteractive add-apt-repository ppa:beineri/opt-qt-5.12.10-focal -y \\
    && apt update \\
    && apt install -y qt512charts-no-lgpl qt512svg qt512xmlpatterns qt512tools qt512translations qt512graphicaleffects qt512quickcontrols2 qt512wayland qt512websockets qt512serialbus qt512serialport qt512location qt512imageformats qt512script qt512scxml qt512gamepad qt5123d
    
"""
    tmp_str += """# Install hstr
RUN DEBIAN_FRONTEND=noninteractive add-apt-repository ppa:ultradvorka/ppa -y \\
    && apt update \\
    && apt update && apt install -y hstr \\
    && rm -rf /var/lib/apt/lists/*
"""
    return tmp_str


def set_entrypoint():
    return """
ENTRYPOINT ["/bin/bash"]
"""


def set_user_permissions():
    return """# >>> Change user permissions >>>
ARG USER_NAME=rangenet
# Set the password '123' for the 'rangenet' user
RUN useradd ${USER_NAME} -m -G sudo -u 1000 -s /bin/bash && echo ${USER_NAME}:123 | chpasswd
USER ${USER_NAME}
"""


def set_workdir():
    return """WORKDIR /home/${USER_NAME}"""


def set_bashrc():
    tmp_str = ""
    if ROS_VERSION == "noetic":
        tmp_str += 'RUN echo "source /opt/qt512/bin/qt512-env.sh" >> ~/.bashrc'

    tmp_str += f"""
RUN echo "source /opt/ros/{ROS_VERSION}/setup.bash" >> ~/.bashrc
"""

    return tmp_str


def download_libtorch():
    return """# >>> Download libtorch >>>
RUN cd ~ \\
    && wget -c https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcu111.zip \\
    && unzip libtorch-cxx11-abi-shared-with-deps-1.10.0+cu111.zip \\
    && rm libtorch-cxx11-abi-shared-with-deps-1.10.0+cu111.zip \\
    && echo "export Torch_DIR=${HOME}/libtorch/share/cmake/Torch" >> ~/.bashrc
    """


def setup_workspace():
    return """
RUN git clone https://github.com/Natsu-Akatsuki/RangeNet-TensorRT ~/workspace/rangenet/src
"""


def generate_dockerfile():
    target_path = target_dir / "Dockerfile"
    with open(target_path, "w") as f:
        f.write(f"FROM {set_basic_image()}\n")
        f.write(f"{set_apt()}\n")
        f.write(f"{set_ros()}\n")
        f.write(f"{set_nvidia()}\n")
        f.write(f"{set_tensort()}\n")
        f.write(f"{set_shell()}\n")
        f.write(f"{set_others()}\n")
        f.write(f"{set_entrypoint()}\n")
        f.write(f"{set_user_permissions()}\n")
        f.write(f"{set_workdir()}\n")
        f.write(f"{set_bashrc()}\n")
        f.write(f"{download_libtorch()}\n")
        f.write(f"{setup_workspace()}\n")


def set_docker_compose():
    user_home = os.getenv("HOME")
    tmp_str = f"""services:
  liobench:
    image: rangenet:{prefix}
    container_name: rangenet
    volumes:
      # Map host workspace to container workspace (Please change to your own path)
      - {user_home}/rangenet:/home/rangenet/workspace/:rw
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - DISPLAY=$DISPLAY
      - NVIDIA_VISIBLE_DEVICES=all
    tty: true
    stdin_open: true
    restart: 'no'
"""
    return tmp_str


def generate_docker_compose():
    target_path = target_dir / "docker-compose.yml"
    with open(target_path, "w") as f:
        f.write(f"{set_docker_compose()}")


def set_build_bash():
    tmp_str = f"""#!/bin/bash
docker build -t rangenet:{prefix} --network=host -f Dockerfile .
"""

    return tmp_str


def generate_build_bash():
    target_path = target_dir / "build_image.sh"
    with open(target_path, "w") as f:
        f.write(f"{set_build_bash()}")


if __name__ == "__main__":
    generate_dockerfile()
    generate_docker_compose()
    generate_build_bash()
