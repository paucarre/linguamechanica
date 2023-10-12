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

RUN sudo apt-get install wget gpg && \
    wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg && \
    sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg && \
    sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list' && \
    rm -f packages.microsoft.gpg && \
    sudo apt install -y apt-transport-https && \
    sudo apt update && \
    sudo apt install -y code && \
    sudo apt-get autoclean

RUN sudo DEBIAN_FRONTEND=noninteractive  apt install -y intel-mkl && \
    echo 'deb [trusted=yes] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list && \
    sudo apt-get update -y && \
    sudo apt-get install -y nvhpc-22-1 && \
    sudo rm -rf /etc/alternatives/cuda && \
    sudo rm -rf /usr/local/cuda && \
    sudo ln -s /opt/nvidia/hpc_sdk/Linux_x86_64/22.1/cuda/11.5  /usr/local/cuda && \
    sudo cp -ans /opt/nvidia/hpc_sdk/Linux_x86_64/22.1/math_libs/11.5/targets/x86_64-linux/include/* /opt/nvidia/hpc_sdk/Linux_x86_64/22.1/cuda/11.5/targets/x86_64-linux/include/ && \
    sudo cp -ans /opt/nvidia/hpc_sdk/Linux_x86_64/22.1/math_libs/11.5/targets/x86_64-linux/lib/* /opt/nvidia/hpc_sdk/Linux_x86_64/22.1/cuda/11.5/targets/x86_64-linux/lib/ && \
    pip3 uninstall -y tensorflow && sudo chmod -R o+w /usr/bin/ && pip3 install --upgrade numpy && \
    sudo apt-get autoclean


RUN cd /home/ros/ &&\
    git clone --branch v2.50.0 https://github.com/IntelRealSense/librealsense.git && \
    cd librealsense && \
    ./scripts/setup_udev_rules.sh && \
    mkdir build && cd build && \
    cmake .. -DBUILD_EXAMPLES=true -DCMAKE_BUILD_TYPE=release -DFORCE_RSUSB_BACKEND=false -DBUILD_WITH_CUDA=false && make -j$(($(nproc)-1))

RUN cd /home/ros/librealsense/build && sudo make install
RUN sudo usermod -a -G video ros
RUN cd /home/ros/librealsense && \
    sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/ && \
    sudo /lib/systemd/systemd-udevd --daemon && \
    sudo udevadm control --reload-rules && udevadm trigger

RUN sudo apt-get update -y && \
    sudo apt-get install -y libnlopt-guile0 libnlopt-dev libnlopt0 libnlopt-cxx0 libnlopt-cxx-dev && \
    sudo apt-get autoclean

RUN cd /home/ros && \
    wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip && \
    rm -rf libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip

ENV Torch_DIR=/home/ros/libtorch

#RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash && \
#    source ~/.bashrc && \
#    nvm install v16.20.2 && \
#    npm install --global yarn && \
#    npm install -g npm@3.7.5 && \
#    yarn global add code-server && \




