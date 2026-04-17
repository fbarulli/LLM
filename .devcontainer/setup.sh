#!/bin/bash

PROJECT_ROOT=$(pwd)

if ! python3 -c "import sys; exit(0 if sys.version.startswith('${PYTHON_VER}') else 1)" 2>/dev/null; then
    echo "ERROR: Python version ${PYTHON_VER} not found."
    exit 1
fi
git config --global --add safe.directory "$PROJECT_ROOT"
if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
fi
python -m venv .venv
. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
. .venv/bin/activate && python -m ipykernel install --user --name venv --display-name "Python ${PYTHON_VER} (venv)"
sed -i '/.venv\/bin\/activate/d' ~/.bashrc
echo "source $PROJECT_ROOT/.venv/bin/activate" >> ~/.bashrc
