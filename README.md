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
    - As `IK` is a perfect-information game, the geodesic loss is used directly to train the actor.
    - There is a `Q-Learning pair` network to deal with cases where the robot trajectory does
    not create monotonically decreasing rewards (the robot increases the geodesic distance to the target
    before getting closer until the `IK` problem is solved) and potential extensions of the project such as grasping.
 - Coordinate systems are continuous. In practical terms, all `so(3)` pose elements and robot `theta` parameters 
 are represented as a (`sin`, `cos`) pairs ( angles don't suffer from discontinuities at zero and 2π ).



# Setup
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

# References
 - [A micro Lie theory for state estimation in robotics](https://arxiv.org/pdf/1812.01537.pdf)
 - [Modern Robotics: Mechanics, Planning, and Control](http://hades.mech.northwestern.edu/index.php/Modern_Robotics)
