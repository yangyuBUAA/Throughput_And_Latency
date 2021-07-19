import logging
logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class ShortTextDataset(Dataset):
    def __init__(self, tokenized_list, label_list):
        self.tokeized_list = tokenized_list
        self.label_list = label_list

    def __getitem__(self, item):
        return self.tokeized_list[item]["input_ids"],\
            self.tokeized_list[item]["attention_mask"],\
            self.tokeized_list[item]["token_type_ids"],\
            torch.tensor(int(self.label_list[item]))

    def __len__(self):
        return len(self.tokeized_list)


def get_short_text_dataset(config, tokenizer):
    train_set = load_dataset(config, tokenizer)
    return train_set


def load_dataset(config, tokenizer):
    CURRENTDIR = config["CURRENT_DIR"]
    # TRAIN_CACHED_PATH = os.path.join(CURRENTDIR, "data/train/train_cached.bin")
    # EVAL_CACHED_PATH = os.path.join(CURRENTDIR, "data/eval/eval_cached.bin")
    # TEST_CACHED_PATH = os.path.join(CURRENTDIR, "data/test/test_cached.bin")
    #
    # if os.path.exists(TEST_CACHED_PATH):
    #     torch.load()

    TRAIN_SOURCE_PATH = os.path.join(CURRENTDIR, config["DATA_DIR"])
    logger.info("构建数据集...")
    train_set = construct_dataset(config, tokenizer, TRAIN_SOURCE_PATH)
    logger.info("构建完成...{}条数据...".format(len(train_set)))
    return train_set

def construct_dataset(config, tokenizer, SOURCE_PATH):
    tokenized_list = list()
    label_list = list()
    with open(SOURCE_PATH, "r", encoding="utf-8") as train_set:
        data = train_set.readlines()
        for line in data:
            line = line.strip()
            sequence, label = line[:-2], line[-1]
            # print(sequence, label)
            # break
            sequence_tokenized = tokenizer(sequence, return_tensors="pt", max_length=config["max_length"], padding="max_length",
                                           truncation=True)
            tokenized_list.append(sequence_tokenized)
            label_list.append(label)

    return ShortTextDataset(tokenized_list, label_list)