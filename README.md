# Hashi bot

This repo contains the main python files that shold be run independantly.

## Dependencies

 - Mac with M1 Pro
 - Ollama - https://ollama.ai/
 - Github access
 - Python 3.11

## Installation

export TOKENIZERS_PARALLELISM=true

1. Clone the repository to your local machine
2. create a Python virtual environment:
```bash
$ python -m venv .venv
$ source .venv/bin/activate
```

3. Install the required dependencies using pip:
```bash
$ pip install -r requirements.txt
```

4. Pull the required model files with ollama (can be changed in `hashi_chat.py` from line 44 onwards):
```bash
$ ollama pull mistral:7b-instruct-v0.2-q4_0
```
* Feel free to change the model used in `hashi_chat.py` line 48

5. Set some environment variables:
```bash
export TOKENIZERS_PARALLELISM=true
```

## Usage

1. create the folder with all the latest documents:
```bash
$ python generate_docs.py
```
2. run the bot:
```bash
$ python hashi_chat.py
```
3. If all runs well you should be able to run chainlit:
```bash
$ python -m chainlit RAG.py
```

These commands are all under `run.sh`

## Creating the embeddings

The quality of the answers only goes as far as the quality of the LLM, prompt and embeddings.
To tweak and create new emebeddings, use the `embed_hashicorp.py` script.

