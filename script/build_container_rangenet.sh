#!/bin/bash
set -e

XAUTH=${HOME}/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]
    then
        # hide the message "tmp/.docker.xauth does not exist"
        echo $xauth_list | xauth -f $XAUTH nmerge - 2> /dev/null
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

# 参数配置
set_container_name="--name=rangenet1.0"
image_name="rangenet:1.0"

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
    
    

