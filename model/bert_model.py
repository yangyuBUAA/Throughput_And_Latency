import os
import torch
from transformers import BertTokenizer, BertModel
from model.dataset import get_short_text_dataset
def get_bert_model(config):
    model = BertModel.from_pretrained(os.path.join(config["CURRENT_DIR"], config["model_path"]))
    return model

def get_bert_dataset(config):
    tokenizer = BertTokenizer.from_pretrained(os.path.join(config["CURRENT_DIR"], config["model_path"]))
    return get_short_text_dataset(config, tokenizer)