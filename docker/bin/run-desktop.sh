#!/bin/bash
mode=$1
if [ -z "$1" ]
  then
    mode="it"
fi
docker container run --rm -$mode \
  --user $(id -u) \
  --name ros-desktop \
  --workdir /home/ros \
  --entrypoint /bin/bash \
  ros-desktop:v0.1