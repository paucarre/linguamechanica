# Lingua Mechanica

![Lingua Mechanica](https://media.giphy.com/media/SyVdwcA3UWGcHV20fS/giphy.gif)

`Lingua Mechanica` solves the `inverse kinematics` problem by training a `Reinforcement Learning` model
from a `URDF` description of a robotic arm. It is able to solve thousands of Inverse Kinematics in parallel
 or create swarms of thousands of initial poses to solve a single inverse kinematics problem.

Specifically, the inverse kinematics solver has the following features:
 - It translates the robot `URDF` into its exponential form, as a differentiable network that outputs forward kinematics.
 - The robot arm kinematic chain is part of the model. An advantage is that the optimizer has access to the robot's jacobian through backpropagation.
 - The `SE(3)` representation are [Implicit Dual Quaternions](http://www.neil.dantam.name/papers/dantam2020robust.pdf) ([Neil T. Dantam](http://www.neil.dantam.name/)), a fast and singularity-free compact `Dual-Quaternion` representation.
 - The `Reinforcement Learning` reward is the cumulative geodesic on `SE(3)`. The geodesic can 
 be weighted if desired ( note that for most collaborative arms the `T(3)` geodesic, in meters, is not far from π ).
 The geodesic loss is formally defined as the (weighted) sum of the vee `se(3)` target pose with respect to the current pose.
 The all computations (forward kinematics, exponentiations, logarithms, etc) are differentiable and integrated into `PyTorch`'s `autograd`.
 - The actor training is two-folded: 
    - The geodesic loss is used directly to train the actor.
    - There is a `Q-Learning pair` network.


# User's Manual

## Setup
Currently only **Ubuntu** is supported, but **PR**s to support other Linux distributions
are welcomed but active efforts will be put on dockerizing the solution.

There are dependencies such as `pyenv` and several other `Ubuntu` dependencies that are required.
This script installs them:
```bash
source ./bin/install-deps.sh
```

The next step is to set up the environment by running run:
```bash
poetry install
poetry shell
```

And finally, with the activated `poetry shell`, install `torch` and other `torch`-related dependencies running:
```bash
source ./bin/install-torch.sh
```

To run unit tests:
```bash
make test
```

## Usage

To train inverse kinematics use:
```bash
python -m linguamechanica.train --urdf URDF_PATH
```

To train inverse kinematics from checkpoint use:
```bash
python -m linguamechanica.train --urdf URDF_PATH --checkpoint CHECKPOINT_ID
```

To run inference use:
```bash
python -m linguamechanica.inference --checkpoint 927000 --target_thetas 0.4,-0.6,0.3,-0.5,0.5,0.2 --iterations 100 --samples 10000
```

To visually test inverse kinematics from checkpoint use:
```bash
python -m linguamechanica.test --urdf URDF_PATH --checkpoint CHECKPOINT_ID
```

Example to visually  test inverse kinematics using `target pose`:
```bash
python -m linguamechanica.test --checkpoint 927000 --target_pose 1.13,-0.935,-0.0869,0.466,-2.67,2.2
```

Example to visually  test inverse kinematics using `target thetas`:
```bash
python -m linguamechanica.test --checkpoint 927000 --target_thetas 0.5,-0.8,0.3,-0.5,0.5,0.6 
```

# Checkpoints

You can find checkpoints in [Google Drive](https://drive.google.com/drive/folders/10r1h3-qMSE0tlQM2KHpXJWahhaNaPg9w?usp=sharing), currently only for CR5 robot arm.
You need to download the checkpoints and leave them into the `checkpoints` folder.
Then launch any of the command line programs such as:
```bash
python -m linguamechanica.test --checkpoint 927000 --target_thetas 0.4,-0.6,0.3,-0.5,0.5,0.2 --iterations 100 --samples 10000
```

# Docker

See the [docker documentation](docker/README.md) to setup a docker.
Note that the way Docker is approached for `Linugua Mechanica` is to 
set up a docker image with a mapped local home folder.
Dockerization of the project is Work In Progress.

# ROS
First step is installing dependencies using the ROS `Python` by running:
```
source ros/bin/install-python-deps.sh
```
Then add the following projects in your ROS workspace while adding `linguamechanica` root folder in `PYTHONPATH` :
```
https://github.com/paucarre/lingua_mechanica_kinematics_server
https://github.com/paucarre/lingua_mechanica_kinematics_msgs
https://github.com/paucarre/lingua_mechanica_kinematics_plugin
```
The next step is to launch the IK server by launching:
```
rosrun lingua_mechanica_kinematics_server lingua_mechanica_kinematics_server.py
```
Finally, add `lingua mechanica` as kinematics solver by updating your IK node configuration:
```
YOUR_ROBOT_ARM:
  kinematics_solver: lingua_mechanica_kinematics_plugin/LinguaMechanicaKinematicsPlugin
```

# References
 - [Robust and efficient forward, differential, and inverse kinematics using dual quaternions](http://www.neil.dantam.name/papers/dantam2020robust.pdf). [Neil T. Dantam](http://www.neil.dantam.name/)
 - [A micro Lie theory for state estimation in robotics](https://arxiv.org/pdf/1812.01537.pdf)
 - [Modern Robotics: Mechanics, Planning, and Control](http://hades.mech.northwestern.edu/index.php/Modern_Robotics)
