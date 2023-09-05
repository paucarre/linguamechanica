#!/bin/bash
eval "$(pyenv init -)"
python -m keyring --disable
pyenv global 3.10.12
poetry shell