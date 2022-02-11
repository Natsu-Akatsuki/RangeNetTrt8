#!/bin/bash

# for debug
# set -x 
if [ ! -d ${HOME}/tmp} ]
then
    mkdir ${HOME}/tmp
fi

if [ ! -d ${HOME}/docker_ws} ]
then
    mkdir ${HOME}/docker_ws
fi

XAUTH=${HOME}/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | sudo xauth -f $XAUTH nmerge -

# 参数配置
set_container_name="--name=rangenet1.0"
image_name="registry.cn-hangzhou.aliyuncs.com/gdut-iidcc/rangenet:1.0"

# 文件挂载
set_volumes="--volume=${HOME}/docker_ws:/docker_ws:rw"

# 开启端口
# pycharmPORT="-p 31111:22" 
# jupyterPORT="-p 8888:8888" 
# tensorboardPORT="-p 6006:6006" 
set_network="--network=host"

# 设备限制
set_shm="--shm-size=8G"

docker run -it --gpus all \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --volume="${HOME}/tmp/.X11-unix:${HOME}/tmp/.X11-unix:rw" \
    --privileged \
    ${set_volumes} \
    ${set_network} \
    ${set_shm} \
    ${set_container_name} \
    $image_name
    
    

