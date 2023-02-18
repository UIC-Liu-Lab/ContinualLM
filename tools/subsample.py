import json
import os.path
import random

import jsonlines
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer



new_data = {}
with open('../data/realnews.jsonl', 'r+') as f, open('../data/format_realnews_small.txt', 'w') as f_sub:
    for item_id, item in enumerate(jsonlines.Reader(f)):
        f_sub.writelines(' '.join(item['title'].split()) + ' ' + ' '.join(item['text'].split()) + '\n')
        if item_id > 1e6:
            break
