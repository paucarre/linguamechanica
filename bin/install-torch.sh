#!/bin/bash
# There is no way for now to do this using poetry in a simple way!!
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
pip3 install "torchrl-nightly[dm_control,gym_continuous,rendering,tests,utils]"
pip3 install https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu113_pyt1110/pytorch3d-0.7.2-cp310-cp310-linux_x86_64.whl
# This is to patch Pytorch 3D with a SE(3) logarithm fix
python -c 'import pytorch3d; print(f"wget https://raw.githubusercontent.com/paucarre/pytorch3d/se3_log_map_fix/pytorch3d/transforms/se3.py -O  {pytorch3d.__path__[0]}/transforms/se3.py")' | bash
python -c 'import pytorch3d; print(f"wget https://raw.githubusercontent.com/paucarre/pytorch3d/se3_log_map_fix/pytorch3d/transforms/so3.py -O  {pytorch3d.__path__[0]}/transforms/so3.py")' | bash
# Torch Metrics
pip install torchmetrics

python -c 'import pytrasform3d; print(pytrasform3d.__path__[0])'