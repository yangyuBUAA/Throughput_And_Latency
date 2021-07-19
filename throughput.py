import time
import yaml

import logging
logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from model.albert_model import get_albert_model, get_albert_dataset
from model.bert_model import get_bert_model, get_bert_dataset
# from model.electra_model import get_electra_model, get_electra_dataset

from torch.utils.data import DataLoader

def run_test(config):
    model_name = config["model_name"]

    max_length = config["max_length"]
    
    if model_name == "albert":
        
        model = get_albert_model(config)
        dataset = get_albert_dataset(config)

        logger.info("albert model!")
        logger.info("******************")
        logger.info("max_length:{}".format(str(max_length)))

    elif model_name == "bert":
        
        model = get_bert_model(config)
        dataset = get_bert_dataset(config)

        logger.info("bert model!")
        logger.info("******************")
        logger.info("max_length:{}".format(str(max_length)))

    elif model_name == "electra":
        get_electra_model(config)
        get_electra_dataset(config)

    model = model.cuda()

    # 存储batch_size从1到100的推理时间
    store_1_100 = list()
    for i in range(2, 100, 2):
        t_load_data = time.time()
        dataloader = DataLoader(dataset, batch_size=i)
        t_load_data_finish = time.time()

        # logger.info("load_data_use_time:{}".format(str(t_load_data_finish-t_load_data)))

        for batch in dataloader:
            t1 = time.time()
            model(batch[0].squeeze().cuda(), batch[1].squeeze().cuda(), batch[2].squeeze().cuda())
            t2 = time.time()
            break
        store_1_100.append(t2-t1)

    store_1_1002time = {index*2+2:store_1_100[index] for index in range(len(store_1_100))}
    logger.info("（100）以内batch_size大小对应的推理时间：")
    logger.info(store_1_1002time)
    # 创建记录，记录最大的batch_size
    record = 100
    # batch增大100条数据时存储一下计算时间
    time_store = list()
    time_store_thres = 100
    
    try:
        while True:
            dataloader = DataLoader(dataset, batch_size=record)
            for batch in dataloader:
                # print(batch[0], batch[1], batch[2])
                t1 = time.time()
                model(batch[0].squeeze().cuda(), batch[1].squeeze().cuda(), batch[2].squeeze().cuda())
                break
            print(record)
            record = record + 100
            if record % 100 == 0:
                t2 = time.time()
                time_store.append((t2-t1))
    except:
        pass
    

    logger.info("最大batch_size：{}".format(str(record-100)))
    logger.info("batch_size大小对应的推理时间：")
    size2time = {(index+1)*100:time_store[index] for index in range(len(time_store))}
    logger.info(size2time)
    logger.info("吞吐量：{}".format(str(record / time_store[-1])))
    logger.info("******************")

        
    t_end = time.time()
    # except:
    #     pass

if __name__=="__main__":
    config_list = ["config_albert_16.yaml",\
                   "config_albert_32.yaml",\
                   "config_albert_64.yaml",\
                   "config_bert_16.yaml",\
                   "config_bert_32.yaml",\
                   "config_bert_64.yaml"]
    for config_file in config_list:
        with open(config_file, 'r', encoding='utf-8') as f:
            result = f.read()
            config = yaml.load(result)
        run_test(config)