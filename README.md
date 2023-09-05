# Introduction

This projects solves the inverse kinematic problem by training a Reinforce Learning model
from a URDF description of a robotic arm. It translated the URDF into its exponential form, 
which is used by the model to train inverse kinematics.

The project makes extensive use of on-manifold optimization, not only in `SE(3)` but also 
its lie algebra `se(3)` is a conitious manifold representation.


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
