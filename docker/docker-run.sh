#!/bin/bash

cd "$(dirname $0)"

if ! docker version | grep 'linux/amd64' ; then
  echo "Could not find docker."
  exit 1
fi

if ! docker-compose version | grep v2 ; then
  echo "docker-compose v2.x is not installed"
  exit 1
fi


if ! docker run -it  --gpus=all --rm nvidia/cuda:11.4.2-base-ubuntu20.04 nvidia-smi | egrep -e 'NVIDIA.*On' ; then
  echo "Docker could not find your NVIDIA gpu"
  exit 1
fi

if ! docker-compose build ; then
  echo "Error while building"
  exit 1
fi
docker-compose up