#!/bin/sh

python3 -m venv .venv
python3 -m pip install --upgrade pip
source .venv/bin/activate
python3 -m pip install -r requirements.txt
alias norminette=flake8
alias py=python3