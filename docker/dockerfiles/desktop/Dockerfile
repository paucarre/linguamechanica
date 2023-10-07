FROM ros-base:v0.1

ENV DEBIAN_FRONTEND noninteractive

ARG USERNAME=ros
USER $USERNAME
RUN mkdir -p /home/ros/phase4ai/ws/arm/src

USER root

ARG USERNAME=ros
USER $USERNAME
WORKDIR /home/$USERNAME

RUN sudo apt update && \
    sudo apt install -y --no-install-recommends --allow-unauthenticated \
        software-properties-common curl apache2-utils \
        supervisor nginx sudo net-tools zenity xz-utils \
        dbus-x11 x11-utils alsa-utils curl ca-certificates gnupg patch \
        mesa-utils libgl1-mesa-dri ffmpeg tmux libusb* \
        lxde gtk2-engines-murrine gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine arc-theme \
        xvfb x11vnc libcanberra-gtk-module libcanberra-gtk3-module meshlab psmisc python-dev python3-dev \
        vim-tiny sudo firefox-geckodriver ttf-ubuntu-font-family ttf-wqy-zenhei tmuxinator python-dev \
        usbutils joystick cheese libglvnd-dev python3-pip && \
    pip install hidapi && \
    pip install --upgrade setuptools && \
    sudo add-apt-repository -r ppa:fcwu-tw/apps && \
    sudo mkdir /usr/local/ffmpeg && \
    sudo apt-get autoclean

USER root

RUN usermod  -a -G video ros

USER $USERNAME

RUN sudo ln -s /usr/bin/ffmpeg /usr/local/ffmpeg/ffmpeg  && \
    sudo chmod -R o+rw /usr/local

RUN pip install catkin_pkg lark PyYAML

#KEEP THIS HERE, DO NOT MOVE THIS UP
RUN sudo apt update && \
    sudo apt install -y python-is-python3 && \
    sudo apt-get autoclean

ENV HOME=/home/ros \
    SHELL=/bin/bash \
    CUDA_HOME=/usr/local/cuda

RUN python -m pip install --user --upgrade pip && \
    sudo apt update && \
    sudo apt install -y python3-send2trash && \
    sudo apt install -y --no-install-recommends iputils-ping ros-noetic-rviz-visual-tools ros-noetic-ddynamic-reconfigure* && \
    sudo apt install -y ros-noetic-geometric-shapes ros-noetic-moveit-msgs \
        ros-noetic-srdfdom ros-noetic-pybind11-catkin ros-noetic-moveit-resources-* \
        ros-noetic-ompl ros-noetic-warehouse-ros-* ros-noetic-eigenpy \
        ros-noetic-moveit-visual-tools ros-noetic-rosparam-shortcuts \
        ros-noetic-collada-urdf ros-noetic-rospy-message-converter && \
    sudo python -m pip install --ignore-installed setuptools testresources && \    
    sudo apt-get autoclean

RUN pip install lxml
RUN sudo chmod -R o+rw /usr/local

RUN sudo apt-get update && \
    sudo apt-get install -y ros-noetic-trac-ik-kinematics-plugin && \
    sudo apt-get install --no-install-recommends -y \
    libblosc-dev \
    libboost-iostreams-dev \
    libboost-system-dev \
    libboost-system-dev \
    libeigen3-dev && \
    sudo apt install -y ros-noetic-tf2* ros-noetic-pcl-* \
    xorg-dev libglu1-mesa-dev python3-dev libsdl2-dev \
    libc++-7-dev libc++abi-7-dev ninja-build libxi-dev gfortran libtbb-dev \
    libosmesa6-dev libudev-dev autoconf libtool flex libkdtree++-dev && \
    sudo apt-get autoclean