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
import os





seqs = ['seq0/640000samples/','seq1/640000samples/','seq2/640000samples/','seq3/640000samples/','seq4/640000samples/']
seeds = ['2021','111','222','333','444']
baselines = ['adapter_bcl/']


for baseline in baselines:

    with open(baseline.replace('/',''),'w') as baseline_f:
        for seq in seqs:
            f1_seeds = []
            acc_seeds = []
            for seed in seeds:
                f1_path = '/hdd_3/zke4/' +  seq + baseline + 'f1_' + seed
                acc_path = '/hdd_3/zke4/' + seq + baseline + 'acc_' + seed


                with open(f1_path,'r') as f1_f, open(acc_path,'r') as acc_f:
                    f1 = np.mean([float(f1) for f1 in f1_f.readlines()])
                    acc = np.mean([float(acc) for acc in acc_f.readlines()])

                # print('f1: ',f1)
                # print('acc: ',acc)

                f1_seeds.append(f1)
                acc_seeds.append(acc)

            f1_seed = np.mean(f1_seeds)
            acc_seed = np.mean(acc_seeds)


            baseline_f.writelines(str(f1_seed) + '\t' + str(acc_seed) + '\n')
            print('baseline: ', baseline)
            print('f1_seed: ', f1_seed)
            print('acc_seed: ', acc_seed)
