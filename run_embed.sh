#!/bin/bash

ulimit -n 10240
if [ -d "env" ]; then
    echo "Directory 'env' already exists."
else 
    python3 -m venv env
fi
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.streamlit.txt
playwright install
ulimit -n 10240
export ANONYMIZED_TELEMETRY=False

if [ -z "$GITHUB_PERSONAL_ACCESS_TOKEN" ]
then
    echo "Please set the GITHUB_PERSONAL_ACCESS_TOKEN environment variable."
    exit
fi

if [ -z "$1" ]
then
  python create_embeddings.py --only_missing
else
  python create_embeddings.py --base_path $1 --only_missing
fi