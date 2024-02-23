#!/usr/bin/env python3

from ai.hashi_chat import ModelDownloader
downloader = ModelDownloader("")
model_list = downloader.list()
for model in model_list['models']:
    print(f"Updating model {model['name']}")
    downloader.download_model(model['name'], True)
