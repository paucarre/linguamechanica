#!/bin/bash
./bin/run-desktop.sh dt
sleep 5
container_id=$(docker ps -aqf "name=ros-desktop" | tr -d '\n')
echo "Docker container $container_id"
destination_folder="$HOME/docker-dir"
mkdir -p $destination_folder/home/ros
mkdir -p $destination_folder/home/ros/.ssh
docker cp $container_id:/home/ros $destination_folder/home
read -p "Do you wish to copy $HOME/.ssh/ folder $destination_folder/home/ros/.ssh ?" choice
case "$choice" in 
  y|Y ) cp $HOME/.ssh/* $destination_folder/home/ros/.ssh;;
  n|N ) echo "SSH keys not moved to $destination_folder/home/ros/.ssh";;
  * ) echo "invalid";;
esac
docker stop -t 1 $container_id