#!/bin/bash

python3 -m venv env
source env/bin/activate

# export GITHUB_PERSONAL_ACCESS_TOKEN=
pip install -r requirements.txt
ollama pull mistral:7b-instruct-v0.2-q4_0
python generate_docs.py
python -m chainlit run RAG.py