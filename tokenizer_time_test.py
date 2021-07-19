"""
测试tokenizer的时延
"""
import time
from transformers import BertTokenizer

def run_test():
    tokenizer = BertTokenizer.from_pretrained("huggingface_pretrained_model/bert-base-chinese")
    sentences = ["今天天气不错啊，挺好" for i in range(5)]
    t1 = time.time()
    sequence_tokenized = tokenizer(sentences, return_tensors="pt", max_length=16, padding="max_length",
                                           truncation=True)

    t2 = time.time()
    print(t2-t1)

if __name__=="__main__":
    run_test()