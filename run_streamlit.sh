#!/bin/bash

if [ -d "env" ]; then
    echo "Directory 'env' already exists."
else 
    python3 -m venv env
fi
source env/bin/activate

export OLLAMA_HOST=http://127.0.0.1:11434
# export GITHUB_PERSONAL_ACCESS_TOKEN=
pip install --upgrade pip
pip install -r requirements.streamlit.txt
ollama pull mistral:7b
ulimit -n 10240
python -m streamlit run main.py