# Introduction

This projects solves the `inverse kinematics` problem by training a `Reinforcement Learning` model
from a `URDF` description of a robotic arm.

Specifically, the inverse kinematics solver has the following features:
 - It translates the robot `URDF` into its exponential form, as a differentiable network that outputs forward kinematics.
 - The robot arm kinematic chain is part of the model. An advantage is that the optimizer has access to the robot's jacobian through backpropagation.
 - The `Reinforcement Learning` reward is the cumulative geodesic on `SE(3)`. The geodesic can 
 be weighted if desired ( note that for most collaborative arms the `T(3)` geodesic, in meters, is not far from π ).
 The geodesic loss is formally defined as the (weighted) sum of the vee `se(3)` target pose with respect to the current pose.
 The all computations (forward kinematics, exponentiations, logarithms, etc) are differentiable and integrated into `PyTorch`'s `autograd`.
 - The actor training is two-folded: 
    - The geodesic loss is used directly to train the actor.
    - There is a `Q-Learning pair` network.


# Tutorial
Currently only **Ubuntu** is supported, but **PR**s to support other Linux distributions
are welcomed but active efforts will be put on dockerizing the solution.

To set up the environment run:
```bash
source ./bin/install-env.sh
```

To run unit tests:
```bash
make test
```

To train inverse kinematics use:
```bash
python -m linguamechanica.train --urdf URDF_PATH
```

To train inverse kinematics from checkpoint use:
```bash
python -m linguamechanica.train --urdf URDF_PATH --checkpoint CHECKPOINT_ID
```

To test inverse kinematics from checkpoint use:
```bash
python -m linguamechanica.test --urdf URDF_PATH --checkpoint CHECKPOINT_ID
```

Example to test inverse kinematics using `target pose`:
```bash
python -m linguamechanica.test --checkpoint 214000 --target_pose 1.13,-0.935,-0.0869,0.466,-2.67,2.2
```

Example to test inverse kinematics using `target thetas`:
```bash
python -m linguamechanica.test --checkpoint 214000 --target_thetas 0.2,-0.4,0.2,-0.3,0.3,0.6
```


# References
 - [A micro Lie theory for state estimation in robotics](https://arxiv.org/pdf/1812.01537.pdf)
 - [Modern Robotics: Mechanics, Planning, and Control](http://hades.mech.northwestern.edu/index.php/Modern_Robotics)
