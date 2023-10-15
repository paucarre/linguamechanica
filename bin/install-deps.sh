#!/bin/bash
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