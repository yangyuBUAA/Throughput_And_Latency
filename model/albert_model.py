import torch
from transformers import BertModel, BertTokenizer

def get_model(model_path):
    model = BertModel.from_pretrained("")

def get_tokenizer(model_path):