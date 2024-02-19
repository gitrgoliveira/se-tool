#!/bin/bash

if [ -d "env" ]; then
    echo "Directory 'env' already exists."
else 
    python3 -m venv env
fi
source env/bin/activate

# export GITHUB_PERSONAL_ACCESS_TOKEN=
pip install --upgrade pip
pip install -r requirements.streamlit.txt
ollama pull mistral:7b
python -m streamlit run streamlit_chat.py