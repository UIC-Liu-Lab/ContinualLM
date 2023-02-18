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
data = ["camera_unsup", "pubmed_unsup", "phone_unsup", "ai_unsup", "acl_unsup", "restaurant_unsup"]

with open('sequence_10', 'w') as f_random_seq:
    for repeat_num in range(10):
        random.shuffle(data)
        f_random_seq.writelines('\t'.join(data) + '\n')
