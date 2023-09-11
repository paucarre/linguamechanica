#!/bin/bash
#pip uninstall poetry
sudo apt install graphviz
sudo apt-get install libreadline* libreadline*-dev libbz2-dev libssl-dev liblzma-dev libsqlite3-dev
curl https://pyenv.run | bash
echo 'export PATH=$PATH:$HOME/.local/bin/' >> ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
pyenv install 3.10.12
eval "$(pyenv init -)"
pyenv global 3.10.12
python -m pip install poetry==1.5.1
echo 'export PATH="$PYENV_ROOT/versions/3.10.12/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
python -m pip install keyring
python -m keyring --disable
poetry install
poetry shell
# There is no way for now to do this using poetry in a simple way!!
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
pip3 install "torchrl-nightly[dm_control,gym_continuous,rendering,tests,utils]"
pip3 install https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu113_pyt1110/pytorch3d-0.7.2-cp310-cp310-linux_x86_64.whl
# This is to patch Pytorch 3D with a SE(3) logarithm fix
python -c 'import pytorch3d; print(f"wget https://raw.githubusercontent.com/paucarre/pytorch3d/se3_log_map_fix/pytorch3d/transforms/se3.py -O  {pytorch3d.__path__[0]}/transforms/se3.py")' | bash
python -c 'import pytorch3d; print(f"wget https://raw.githubusercontent.com/paucarre/pytorch3d/se3_log_map_fix/pytorch3d/transforms/so3.py -O  {pytorch3d.__path__[0]}/transforms/so3.py")' | bash