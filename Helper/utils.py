import jsonlines
import json
from typing import List
import logging
import torch
import gc
import os
import random
import numpy as np

from .constants import LOGGER_NAME, SEED, CLAIM_CONSTRUCTOR

LOGGER_NAME='MISQA'
SEED = 1024

logger = logging.getLogger(LOGGER_NAME)

def init_logger(verbose: bool = False, log_file: str = ''):
    logging.getLogger().setLevel(logging.DEBUG)
    logger = logging.getLogger(LOGGER_NAME)
    
    if not len(logger.handlers):
        # log.info will always be show in console
        # log.debug will also be shown when verbose flag is set
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.DEBUG if verbose else logging.INFO)

        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

        if log_file != '':
            log_file_dir = '/'.join(log_file.split('/')[:-1])
            make_directory(log_file_dir)
            f_handler = logging.FileHandler(log_file)
            logger.addHandler(f_handler)
            logger.info("file handler added.")
    return logger

def read_json(json_path:str) -> dict:
    with open(json_path, 'r') as f:
        return json.load(f)
    
def read_jsonl(jsonl_path:str) -> List[dict]:
    with jsonlines.open(jsonl_path) as f:
        items = []
        for line in f.iter():
            items.append(line)
    return items

def save_json(json_obj: dict, output_path:str='./data/processed_tweets.json'):
    saving_dir = '/'.join(output_path.split('/')[:-1])
    make_directory(saving_dir)
    jsonFile =  open(output_path, 'w')
    jsonFile.write(json.dumps(json_obj, indent=4, sort_keys=False))
    jsonFile.close()

def save_jsonl(items:List[dict], output_path:str):
    saving_dir = '/'.join(output_path.split('/')[:-1])
    make_directory(saving_dir)
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(items)
        
def jsonl_to_json(items:List[dict]) -> dict:
    json_obj = {}
    for idx, item in enumerate(items):
        json_obj[str(idx)] = item      
    return json_obj

def get_raw(docid:str, corpus_dir:str = './data/tweets/') -> str:
    path = corpus_dir + docid + '.json'
    json_obj = read_json(path)
    return json_obj['contents'].split('\n')[-1]

def get_raw_with_title(docid:str, corpus_dir:str = './data/tweets/') -> str:
    path = corpus_dir + docid + '.json'
    json_obj = read_json(path)
    return json_obj['contents']

def set_SEED():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def set_device():
    if torch.cuda.is_available():
        clear_gpu_cache()
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(SEED)
        logger.info(f"There are {torch.cuda.device_count()} GPU(s) avaiable")
        logger.info(f"We will be using the following GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("No GPU available using the CPU instead")
        device =  torch.device("cpu")
    return device

def clear_gpu_cache() -> None:
    gc.enable()
    torch.cuda.empty_cache()

def build_claim(question: str, answer: str) -> str:
    return question + CLAIM_CONSTRUCTOR + answer

def make_directory(dir: str):
    print(f"dir: {dir}")
    if dir[-1] != '/':
        dir = dir + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir
