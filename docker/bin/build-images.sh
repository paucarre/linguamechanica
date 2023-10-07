#!/bin/bash
docker build -t ros-base:v0.1 --build-arg BUILD_USER_ID=$(id -u) --build-arg BUILD_GROUP_ID=$(id -g) dockerfiles/base
docker build -t ros-desktop:v0.1 dockerfiles/desktop