#!/usr/bin/env python3

from torch import mode
from ai.common import ModelDownloader

downloader = ModelDownloader("")
model_list = downloader.list()
# print (model_list)
for model in model_list['models']:
    print(f"Updating model {model['model']}")
    downloader.download_model(model['model'], True)
