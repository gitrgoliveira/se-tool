
#!/bin/bash


ulimit -n 10240
python3 -m venv env
source env/bin/activate

# export GITHUB_PERSONAL_ACCESS_TOKEN=
pip install -r requirements.streamlit.txt
python embed_hashicorp.py