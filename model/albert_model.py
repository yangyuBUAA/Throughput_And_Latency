import os
import torch
from transformers import BertTokenizer, AutoModel
from model.dataset import get_short_text_dataset
def get_albert_model(config):
    model = AutoModel.from_pretrained(os.path.join(config["CURRENT_DIR"], config["model_path"]))
    return model

def get_albert_dataset(config):
    tokenizer = BertTokenizer.from_pretrained(os.path.join(config["CURRENT_DIR"], config["model_path"]))
    return get_short_text_dataset(config, tokenizer)