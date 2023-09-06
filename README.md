# Introduction

This projects solves the `inverse kinematics` problem by training a `Reinforcement Learning` model
from a `URDF` description of a robotic arm.

Specifically, the inverse kinematics solver has the following features:
 - It translates the robot `URDF` into its exponential form, as a differentiable network that outputs forward kinematics.
 - The robot arm kinematic chain forward kinematics is part of the geodesic error computation
 and thus the whole differentiable chain is embedded into the network. An advantage
 is that the optimizer has access to the robot's jacobian through backpropagation.
 - The `Reinforcement Learning` reward is the cumulative geodesic on `SE(3)`, the geodesic can 
 be weighted if desired ( note that for most collaborative arms the `T(3)` geodesic, in meters, is not far from π ).
 The geodesic loss is computed on `SE(3)` manifold using the vee lograithm of the target pose with respect to the current pose.
 The geodedic loss computation, including the forward kinematics, exponentiations and logarithms, are fully differentiable and integrated into `PyTorch`'s `autograd`.
 - The actor training is two folded: 
    - The perfect-information `IK` game is incorporated into the actor training as geodesic loss.
    - There is a `Q-Learning pair` quality network to deal with cases where the robot trajectory does
    not create monotically decreasing rewards (the robot increases the geodesic distance to the target
    before getting closer until the `IK` problem is solved) and potential extensions of the project such as grasping.
 - All coordinate systems are continious. In practical terms, all angular components of the `se(3)` algebra
 are represented as "`SO(2)` charts" ( angles don't suffer from discontinuities at zero and 2π ).
 


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

- Lie Group Theory
    - [A micro Lie theory for state estimation in robotics](https://arxiv.org/pdf/1812.01537.pdf)
