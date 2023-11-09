#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        realpath -m "$PWD"/"$1"
    fi
}

ARCH=`uname -m`
if [ $ARCH == "x86_64" ]; then
    if command -v nvidia-smi &> /dev/null; then
        DEVICE=cuda
        ARGS="--ipc host --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all"
    else
        echo "${orange}CPU-only${reset_color} build is not supported yet"
        exit 1
    fi
else
    echo "${orange}${ARCH}${reset_color} architecture is not supported"
    exit 1
fi

if [ $# != 1 ]; then
    echo "Usage:
          bash start.sh [DATA_DIR]
        "
    exit 1
fi

DATA_DIR=$(get_real_path "$1")

if [ ! -d $DATA_DIR ]; then
    echo "Error: DATA_DIR=$DATA_DIR is not an existing directory."
    exit 1
fi

PROJECT_ROOT_DIR=$(cd ./"`dirname $0`"/.. || exit; pwd)

echo "Running on ${orange}${ARCH}${reset_color} with ${orange}${DEVICE}${reset_color}"

docker run -it -d --rm \
    $ARGS \
    --privileged \
    --name ${USER}_oneformer \
    --net host \
    -v $PROJECT_ROOT_DIR:/home/docker_oneformer/OneFormer:rw \
    -v $DATA_DIR:/home/docker_oneformer/data:rw \
    oneformer:$DEVICE-$USER

docker exec --user root \
    ${USER}_oneformer bash -c "/etc/init.d/ssh start"
