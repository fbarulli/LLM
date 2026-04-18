#!/bin/bash

# Cleanup legacy venv artifacts
rm -rf .venv

# Git permissions
git config --global --add safe.directory "${containerWorkspaceFolder:-$(pwd)}"

# Env setup
[ -f .env.example ] && [ ! -f .env ] && cp .env.example .env

# Force kernel registration to the specific 3.11.15 path
/usr/local/bin/python3 -m ipykernel install --user --name python3 --display-name "Python 3.11 (System)"
