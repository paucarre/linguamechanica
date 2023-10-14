#!/bin/bash
pip install --user -r requirements.txt
pip install --user "torchrl-nightly[dm_control,gym_continuous,rendering,tests,utils]"
pip3 install --user https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/pytorch3d-0.7.4-cp38-cp38-linux_x86_64.whl
# This is to patch Pytorch 3D with a SE(3) logarithm fix
python -c 'import pytorch3d; print(f"wget https://raw.githubusercontent.com/paucarre/pytorch3d/se3_log_map_fix/pytorch3d/transforms/se3.py -O  {pytorch3d.__path__[0]}/transforms/se3.py")' | bash
python -c 'import pytorch3d; print(f"wget https://raw.githubusercontent.com/paucarre/pytorch3d/se3_log_map_fix/pytorch3d/transforms/so3.py -O  {pytorch3d.__path__[0]}/transforms/so3.py")' | bash
# Torch Metrics
pip install torchmetrics
export PYTHONPATH=$PYTHONPATH:$HOME/linguamechanica
