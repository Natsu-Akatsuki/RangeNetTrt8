from pathlib import Path
import argparse
import os
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Dockerfile and related files.")
    parser.add_argument("--ubuntu_version", default="22.04", choices=["22.04", "20.04"], help="Ubuntu version (default: 22.04)")
    parser.add_argument("--ros_version", default="humble", choices=["noetic", "humble"], help="ROS version (default: humble)")
    parser.add_argument("--tensorrt_version", default="10.6", help="TensorRT version (default: 10.6)")
    parser.add_argument("--cuda_version", default="12.4.1", help="CUDA version (default: 12.4.1)")

    return parser.parse_args()


def generate_dockerfile(ubuntu_version, ros_version, tensorrt_version, cuda_version, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)

    def set_basic_image():
        if cuda_version == "11.1.1":
            return f"nvidia/cuda:{cuda_version}-cudnn8-devel-ubuntu{ubuntu_version}"
        else:
            return f"nvidia/cuda:{cuda_version}-cudnn-devel-ubuntu{ubuntu_version}"

    def set_ros():
        # Note: the default bash is /bin/sh, thus we do not use "source"
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
RUN curl --resolve raw.githubusercontent.com:443:185.199.108.133 -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \\
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \\
    && apt update \\
    && DEBIAN_FRONTEND=noninteractive apt install -y ros-humble-desktop python3-colcon-common-extensions ros-humble-pcl-ros
"""
        }
        return tmp_dict[ros_version]

    def set_apt():
        return """
RUN sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list \\
    && rm /etc/apt/sources.list.d/* \\
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
        sudo \\
        unzip \\
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
        tmp_str += f"ENV TENSORRT_VERSION {tensorrt_version}\n"
        if ubuntu_version == "20.04":
            tmp_str += """RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub \\
    && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" \\
"""
        if ubuntu_version == "22.04":
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
        if ubuntu_version == "20.04":
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
        if ros_version == "noetic":
            tmp_str += 'RUN echo "source /opt/qt512/bin/qt512-env.sh" >> ~/.bashrc'

        tmp_str += f"""
RUN echo "source /opt/ros/{ros_version}/setup.bash" >> ~/.bashrc
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
RUN git clone https://github.com/Natsu-Akatsuki/RangeNet-TensorRT ~/rangenet/rangenet/src
"""

    def download_pcl():
        return """# Fix https://github.com/PointCloudLibrary/pcl/pull/5252
RUN cd ~ \\
    && wget -c https://github.com/PointCloudLibrary/pcl/releases/download/pcl-1.13.1/source.zip \\
    && unzip source.zip \\
    && rm source.zip \\
    && cd pcl \\
    && mkdir build \\
    && cd build \\
    && cmake -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_visualization=ON -DBUILD_apps=OFF -DBUILD_examples=OFF -DBUILD_tools=OFF -DBUILD_samples=OFF .. \\
    && make -j4 \\
    && echo "123" | sudo -S make install \\
    && cd ~ \\
    && rm -rf pcl
"""

    def download_cmake():
        return """# >>> Download CMake >>>
RUN pip3 install --user cmake==3.18 \\
    && echo 'export PATH=${HOME}/.local/bin:${PATH}' >> ~/.bashrc
"""

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
        # f.write(f"{setup_workspace()}\n")
        if ubuntu_version == "22.04":
            f.write(f"{download_pcl()}\n")
        if ubuntu_version == "20.04":
            f.write(f"{download_cmake()}\n")


def set_docker_compose(prefix):
    return f"""services:
  liobench:
    image: rangenet:{prefix}
    container_name: rangenet
    volumes:
      # Map host workspace to container workspace (Please change to your own path)
      - $HOME/rangenet:/home/rangenet/rangenet/:rw
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
      - LANG=C.UTF-8 # Fix garbled Chinese text
      - LC_ALL=C.UTF-8
    tty: true
    stdin_open: true
    restart: 'no'
"""


def generate_docker_compose(target_dir, prefix):
    target_path = target_dir / "docker-compose.yml"
    with open(target_path, "w") as f:
        f.write(f"{set_docker_compose(prefix)}")


def set_build_bash(prefix):
    return f"""#!/bin/bash
docker build -t rangenet:{prefix} --network=host -f Dockerfile .
"""


def generate_build_bash(target_dir, prefix):
    target_path = target_dir / "build_image.sh"
    with open(target_path, "w") as f:
        f.write(f"{set_build_bash(prefix)}")


def generate_build_image_workflow(prefix):
    template_path = Path(__file__).resolve().parent / "build_image_template.yml"
    with open(template_path, 'r') as file:
        content = file.read()

    content = re.sub(r'\${{ PREFIX }}', prefix, content)
    target_path = Path(__file__).resolve().parent.parent / '.github/workflows' / f"build_image_{prefix}.yml"
    with open(target_path, 'w') as file:
        file.write(content)


def main():
    args = parse_args()

    if 1:
        # {22.04, 20.04}
        ubuntu_version = "20.04"
        # {noetic, humble}
        ros_version = "noetic"
        tensorrt_version = "10.6"
        # {11.1.1, 12.4.1 (require: nvidia-driver 550)}
        cuda_version = "11.1.1"
    else:
        ubuntu_version = args.ubuntu_version
        ros_version = args.ros_version
        tensorrt_version = args.tensorrt_version
        cuda_version = args.cuda_version

    target_dir = Path(__file__).resolve().parent
    prefix = f"ubuntu{ubuntu_version}_cuda{cuda_version}_tensorrt{tensorrt_version}_{ros_version}"
    target_dir = target_dir / prefix

    generate_dockerfile(ubuntu_version, ros_version, tensorrt_version, cuda_version, target_dir)
    generate_docker_compose(target_dir, prefix)
    generate_build_bash(target_dir, prefix)
    generate_build_image_workflow(prefix)


if __name__ == "__main__":
    main()
