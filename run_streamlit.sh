#!/bin/bash

python3 -m venv env
source env/bin/activate

# export GITHUB_PERSONAL_ACCESS_TOKEN=

pip install -r requirements.streamlit.txt
ollama pull mistral:7b
python -m streamlit run streamlit.py