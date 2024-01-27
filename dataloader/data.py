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

implemented_datasets = [
    'restaurant_unsup',
    'acl_unsup',
    'phone_unsup',
    'camera_unsup',
    'ai_unsup',
    'pubmed_unsup',
    'restaurant_sup',
    'chemprot_sup',
    'aclarc_sup',
    'scierc_sup',
    'wikitext_unsup'
]
implemented_datasets += [f'hoc{i}_sup' for i in range(10)]

dataset_class_num = {
    'restaurant_sup': 3,
    'chemprot_sup': 13,
    'aclarc_sup': 6,
    'scierc_sup': 7,
    'phone_sup': 2,
    'camera_sup': 2,
}


def get_camera_unsup(args):
    data_files = {'train': './data/absa/post_train/camera.txt'}
    datasets = load_dataset('text', data_files=data_files)
    datasets["validation"] = load_dataset(
        'text', data_files=data_files,split=f"train[:{args.validation_split_percentage}%]"
    )
    datasets["train"] = load_dataset(
        'text', data_files=data_files,
        split=f"train[{args.validation_split_percentage}%:]",
    )
    return datasets


def get_phone_unsup(args):
    data_files = {'train': './data/absa/post_train/phone.txt'}
    datasets = load_dataset('text', data_files=data_files)
    datasets["validation"] = load_dataset(
        'text', data_files=data_files,split=f"train[:{args.validation_split_percentage}%]"
    )
    datasets["train"] = load_dataset(
        'text', data_files=data_files,
        split=f"train[{args.validation_split_percentage}%:]",
    )
    return datasets


def get_restaurant_unsup(args):
    data_files = {'train': './data/yelp_restaurant.txt'}
    datasets = load_dataset('text', data_files=data_files)
    datasets["validation"] = load_dataset(
        'text', data_files=data_files,split=f"train[:{args.validation_split_percentage}%]"
    )
    datasets["train"] = load_dataset(
        'text', data_files=data_files,
        split=f"train[{args.validation_split_percentage}%:]",
    )
    return datasets


def get_acl_unsup(args):
    data_files = {'train': './data/acl_anthology.txt'}
    datasets = load_dataset('text', data_files=data_files)
    datasets["validation"] = load_dataset(
        'text', data_files=data_files,split=f"train[:{args.validation_split_percentage}%]"
    )
    datasets["train"] = load_dataset(
        'text', data_files=data_files,
        split=f"train[{args.validation_split_percentage}%:]",
    )
    return datasets


def get_ai_unsup(args):
    data_files = {'train': './data/ai_corpus.txt'}
    datasets = load_dataset('text', data_files=data_files)
    datasets["validation"] = load_dataset(
        'text', data_files=data_files,split=f"train[:{args.validation_split_percentage}%]"
    )
    datasets["train"] = load_dataset(
        'text', data_files=data_files,
        split=f"train[{args.validation_split_percentage}%:]",
    )
    return datasets



def get_pubmed_unsup(args):
    data_files = {'train': './data/format_pubmed_small.txt'}  # 989 Mb
    datasets = load_dataset('text', data_files=data_files)
    datasets["validation"] = load_dataset(
        'text', data_files=data_files,split=f"train[:{args.validation_split_percentage}%]"
    )
    datasets["train"] = load_dataset(
        'text', data_files=data_files,
        split=f"train[{args.validation_split_percentage}%:]",
    )
    return datasets


def get_wikitext_unsup(args):
    datasets = load_dataset("wikitext", "wikitext-103-v1")
    datasets["validation"] = load_dataset(
        "wikitext","wikitext-103-v1",split=f"train[:{args.validation_split_percentage}%]"
    )
    datasets["train"] = load_dataset(
        "wikitext","wikitext-103-v1",
        split=f"train[{args.validation_split_percentage}%:]",
    )
    return datasets



def get_dataset(dataset_name,tokenizer,args):
    # --- Unsupervised Learning datasets ---
    # attributes: 'text'

    if dataset_name == 'restaurant_unsup':
        datasets = get_restaurant_unsup(args)

    elif dataset_name == 'acl_unsup':
        datasets = get_acl_unsup(args)

    elif dataset_name == 'ai_unsup':
        datasets = get_ai_unsup(args)

    elif dataset_name == 'pubmed_unsup':
        datasets = get_pubmed_unsup(args)


    elif dataset_name == 'camera_unsup':
        datasets = get_camera_unsup(args)

    elif dataset_name == 'phone_unsup':
        datasets = get_phone_unsup(args)

    elif dataset_name == 'wikitext_unsup':
        datasets = get_wikitext_unsup(args)

    # --- Supervised Learning datasets ---
    # attributes: 'text', 'labels'


    elif dataset_name == 'restaurant_sup':

        def label2idx(label):
            if label == 'positive':
                return 0
            elif label == 'neutral':
                return 1
            elif label == 'negative':
                return 2
            else:  # remove contradictive
                print('ignore: ' + label)

        new_data = {}
        for ds in ['train', 'test']:
            new_data[ds] = {}
            new_data[ds]['text'] = []
            new_data[ds]['labels'] = []
            with open(os.path.join('./data/SemEval14-res', ds + '.json')) as f:
                data = json.load(f)
            for _data in data:
                new_data[ds]['text'].append(
                    data[_data]['term'] + ' ' + tokenizer.sep_token + data[_data]['sentence'])  # add aspect as well
                new_data[ds]['labels'].append(label2idx(data[_data]['polarity']))
        datasets = DatasetDict(
            {
                'train': Dataset.from_dict(new_data['train']),
                'test': Dataset.from_dict(new_data['test'])
            }
        )


    elif dataset_name == 'chemprot_sup':


        label2idx = {'DOWNREGULATOR': 0, 'SUBSTRATE': 1, 'INDIRECT-UPREGULATOR': 2, 'INDIRECT-DOWNREGULATOR': 3,
                     'AGONIST': 4, 'ACTIVATOR': 5, 'PRODUCT-OF': 6, 'AGONIST-ACTIVATOR': 7, 'INHIBITOR': 8,
                     'UPREGULATOR': 9, 'SUBSTRATE_PRODUCT-OF': 10, 'AGONIST-INHIBITOR': 11, 'ANTAGONIST': 12}
        new_data = {}

        for ds in ['train', 'test', 'dev']:

            if ds in ['train', 'test']:
                var_ds = ds
                new_data[var_ds] = {}
                new_data[var_ds]['text'] = []
                new_data[var_ds]['labels'] = []
            elif ds == 'dev':
                var_ds = 'test'

            f_ds = ds
            with open('./data/chemprot/{}.jsonl'.format(f_ds), 'r+') as f:
                for item in jsonlines.Reader(f):
                    new_data[var_ds]['text'].append(item['text'])
                    new_data[var_ds]['labels'].append(label2idx[item['label']])



        # we may re-partitial, by classes
        train_ratio = 0.5
        num_label = len(label2idx)
        total_num = len(new_data['train']['labels'])
        print("total_num: ", total_num)
        print("len(new_data['test']['labels']): ", len(new_data['test']['labels']))

        for label in range(num_label):
            num_takeout = int((total_num * (1-train_ratio)) // num_label)
            label_pos = [lab_id for lab_id,lab in enumerate(new_data['train']['labels']) if lab == label][:num_takeout]
            # print('num_takeout: ',num_takeout)
            label_takeout = [lab for lab_id,lab in enumerate(new_data['train']['labels']) if lab_id in label_pos]
            text_takeout = [lab for lab_id,lab in enumerate(new_data['train']['text']) if lab_id in label_pos]
            # print('label_takeout: ',len(label_takeout))

            new_data['test']['labels'] += label_takeout
            new_data['test']['text'] += text_takeout

            new_data['train']['labels'] = [lab for lab_id,lab in enumerate(new_data['train']['labels']) if lab_id not in label_pos]
            new_data['train']['text'] = [lab for lab_id,lab in enumerate(new_data['train']['text']) if lab_id not in label_pos]

            # print("len(new_data['train']['labels']): ",len(new_data['train']['labels']))
            # print("len(new_data['test']['labels']): ",len(new_data['test']['labels']))



        datasets = DatasetDict(
            {
                'train': Dataset.from_dict(new_data['train']),
                'test': Dataset.from_dict(new_data['test'])
            }
        )




    elif dataset_name == 'aclarc_sup':

        label2idx = {'Uses': 0, 'Future': 1, 'CompareOrContrast': 2, 'Motivation': 3, 'Extends': 4, 'Background': 5}
        new_data = {}
        for ds in ['train', 'test', 'dev']:

            if ds in ['train', 'test']:
                var_ds = ds
                new_data[var_ds] = {}
                new_data[var_ds]['text'] = []
                new_data[var_ds]['labels'] = []
            elif ds == 'dev':
                var_ds = 'test'

            f_ds = ds
            with open('./data/citation_intent/{}.jsonl'.format(f_ds), 'r+') as f:
                for item in jsonlines.Reader(f):
                    new_data[var_ds]['text'].append(item['text'])
                    new_data[var_ds]['labels'].append(label2idx[item['label']])


        # we may re-partitial, by classes
        train_ratio = 0.9
        num_label = len(label2idx)
        total_num = len(new_data['train']['labels'])
        print("total_num: ", total_num)
        print("len(new_data['test']['labels']): ", len(new_data['test']['labels']))

        for label in range(num_label):
            num_takeout = int((total_num * (1-train_ratio)) // num_label)
            label_pos = [lab_id for lab_id,lab in enumerate(new_data['train']['labels']) if lab == label][:num_takeout]
            # print('num_takeout: ',num_takeout)
            label_takeout = [lab for lab_id,lab in enumerate(new_data['train']['labels']) if lab_id in label_pos]
            text_takeout = [lab for lab_id,lab in enumerate(new_data['train']['text']) if lab_id in label_pos]
            # print('label_takeout: ',len(label_takeout))

            new_data['test']['labels'] += label_takeout
            new_data['test']['text'] += text_takeout

            new_data['train']['labels'] = [lab for lab_id,lab in enumerate(new_data['train']['labels']) if lab_id not in label_pos]
            new_data['train']['text'] = [lab for lab_id,lab in enumerate(new_data['train']['text']) if lab_id not in label_pos]

            # print("len(new_data['train']['labels']): ",len(new_data['train']['labels']))
            # print("len(new_data['test']['labels']): ",len(new_data['test']['labels']))

        datasets = DatasetDict(
            {
                'train': Dataset.from_dict(new_data['train']),
                'test': Dataset.from_dict(new_data['test'])
            }
        )

    elif dataset_name == 'scierc_sup':



        label2idx = {'FEATURE-OF': 0, 'CONJUNCTION': 1, 'EVALUATE-FOR': 2, 'HYPONYM-OF': 3, 'USED-FOR': 4,
                     'PART-OF': 5, 'COMPARE': 6}
        new_data = {}
        for ds in ['train', 'test', 'dev']:

            if ds in ['train', 'test']:
                var_ds = ds
                new_data[var_ds] = {}
                new_data[var_ds]['text'] = []
                new_data[var_ds]['labels'] = []
            elif ds == 'dev':
                var_ds = 'test'

            f_ds = ds
            with open('./data/sciie/{}.jsonl'.format(f_ds), 'r+') as f:
                for item in jsonlines.Reader(f):
                    new_data[var_ds]['text'].append(item['text'])
                    new_data[var_ds]['labels'].append(label2idx[item['label']])

        # we may re-partitial, by classes
        train_ratio = 0.7
        num_label = len(label2idx)
        total_num = len(new_data['train']['labels'])
        print("total_num: ", total_num)
        print("len(new_data['test']['labels']): ", len(new_data['test']['labels']))

        for label in range(num_label):
            num_takeout = int((total_num * (1-train_ratio)) // num_label)
            label_pos = [lab_id for lab_id,lab in enumerate(new_data['train']['labels']) if lab == label][:num_takeout]
            # print('num_takeout: ',num_takeout)
            label_takeout = [lab for lab_id,lab in enumerate(new_data['train']['labels']) if lab_id in label_pos]
            text_takeout = [lab for lab_id,lab in enumerate(new_data['train']['text']) if lab_id in label_pos]
            # print('label_takeout: ',len(label_takeout))

            new_data['test']['labels'] += label_takeout
            new_data['test']['text'] += text_takeout

            new_data['train']['labels'] = [lab for lab_id,lab in enumerate(new_data['train']['labels']) if lab_id not in label_pos]
            new_data['train']['text'] = [lab for lab_id,lab in enumerate(new_data['train']['text']) if lab_id not in label_pos]

            # print("len(new_data['train']['labels']): ",len(new_data['train']['labels']))
            # print("len(new_data['test']['labels']): ",len(new_data['test']['labels']))



        datasets = DatasetDict(
            {
                'train': Dataset.from_dict(new_data['train']),
                'test': Dataset.from_dict(new_data['test'])
            }
        )

    elif dataset_name == 'hyperpartisan_sup':

        label2idx = {'false': 0, 'true': 1}

        new_data = {}
        for ds in ['train', 'test', 'dev']:

            if ds in ['train', 'test']:
                var_ds = ds
                new_data[var_ds] = {}
                new_data[var_ds]['text'] = []
                new_data[var_ds]['labels'] = []
            elif ds == 'dev':
                var_ds = 'test'

            f_ds = ds
            with open('./data/hyperpartisan_news/{}.jsonl'.format(f_ds), 'r+') as f:
                for item in jsonlines.Reader(f):
                    new_data[var_ds]['text'].append(item['text'])
                    new_data[var_ds]['labels'].append(label2idx[item['label']])

        # we may re-partitial, by classes
        train_ratio = 0.4
        num_label = len(label2idx)
        total_num = len(new_data['train']['labels'])
        print("total_num: ", total_num)
        print("len(new_data['test']['labels']): ", len(new_data['test']['labels']))

        for label in range(num_label):
            num_takeout = int((total_num * (1-train_ratio)) // num_label)
            label_pos = [lab_id for lab_id,lab in enumerate(new_data['train']['labels']) if lab == label][:num_takeout]
            # print('num_takeout: ',num_takeout)
            label_takeout = [lab for lab_id,lab in enumerate(new_data['train']['labels']) if lab_id in label_pos]
            text_takeout = [lab for lab_id,lab in enumerate(new_data['train']['text']) if lab_id in label_pos]
            # print('label_takeout: ',len(label_takeout))

            new_data['test']['labels'] += label_takeout
            new_data['test']['text'] += text_takeout

            new_data['train']['labels'] = [lab for lab_id,lab in enumerate(new_data['train']['labels']) if lab_id not in label_pos]
            new_data['train']['text'] = [lab for lab_id,lab in enumerate(new_data['train']['text']) if lab_id not in label_pos]

            # print("len(new_data['train']['labels']): ",len(new_data['train']['labels']))
            # print("len(new_data['test']['labels']): ",len(new_data['test']['labels']))

        datasets = DatasetDict(
            {
                'train': Dataset.from_dict(new_data['train']),
                'test': Dataset.from_dict(new_data['test'])
            }
        )



    elif dataset_name == 'camera_sup':
        new_data = {}
        label2idx = {'+': 0, '-': 1}
        # add train
        new_data['train'] = {}
        new_data['train']['text'] = []
        new_data['train']['labels'] = []


        for ds in ['train', 'test', 'dev']: #all train
            with open('./data/absa/dat/Bing5Domains/CanonG3/{}.json'.format(ds), 'r') as f:
                tmp_data = json.load(f)
            for dt in tmp_data:
                new_data['train']['labels'].append(label2idx[tmp_data[dt]['polarity']])
                new_data['train']['text'].append(tmp_data[dt]['term'] + ' ' + tokenizer.sep_token + tmp_data[dt]['sentence'])

        new_data['test'] = {}
        new_data['test']['text'] = []
        new_data['test']['labels'] = []

        # if we only consider ood as test...
        # with open('./data/absa/dat/Bing5Domains/CanonG3/{}.json'.format('test'), 'r') as f:
        #     tmp_data = json.load(f)
        # for dt in tmp_data:
        #     new_data['test']['labels'].append(label2idx[tmp_data[dt]['polarity']])
        #     new_data['test']['text'].append(tmp_data[dt]['sentence'])

        for ds in ['train', 'test', 'dev']: # all test
            with open('./data/absa/dat/Bing5Domains/NikonCoolpix4300/{}.json'.format(ds), 'r') as f:
                tmp_data = json.load(f)
            for dt in tmp_data:
                new_data['test']['labels'].append(label2idx[tmp_data[dt]['polarity']])
                new_data['test']['text'].append(tmp_data[dt]['term'] + ' ' + tokenizer.sep_token + tmp_data[dt]['sentence'])

            with open('./data/absa/dat/Bing9Domains/CanonPowerShotSD500/{}.json'.format(ds), 'r') as f:
                tmp_data = json.load(f)
            for dt in tmp_data:
                new_data['test']['labels'].append(label2idx[tmp_data[dt]['polarity']])
                new_data['test']['text'].append(tmp_data[dt]['term'] + ' ' + tokenizer.sep_token + tmp_data[dt]['sentence'])

            with open('./data/absa/dat/Bing9Domains/CanonS100/{}.json'.format(ds), 'r') as f:
                tmp_data = json.load(f)
            for dt in tmp_data:
                new_data['test']['labels'].append(label2idx[tmp_data[dt]['polarity']])
                new_data['test']['text'].append(tmp_data[dt]['term'] + ' ' + tokenizer.sep_token + tmp_data[dt]['sentence'])

        #TODO: ----------

        # we may re-partitial, by classes
        train_ratio = 0.8
        num_label = len(label2idx)
        total_num = len(new_data['train']['labels'])
        print("total_num: ", total_num)
        print("len(new_data['test']['labels']): ", len(new_data['test']['labels']))

        for label in range(num_label):
            num_takeout = int((total_num * (1-train_ratio)) // num_label)
            label_pos = [lab_id for lab_id,lab in enumerate(new_data['train']['labels']) if lab == label][:num_takeout]
            # print('num_takeout: ',num_takeout)
            label_takeout = [lab for lab_id,lab in enumerate(new_data['train']['labels']) if lab_id in label_pos]
            text_takeout = [lab for lab_id,lab in enumerate(new_data['train']['text']) if lab_id in label_pos]
            # print('label_takeout: ',len(label_takeout))

            new_data['test']['labels'] += label_takeout
            new_data['test']['text'] += text_takeout

            new_data['train']['labels'] = [lab for lab_id,lab in enumerate(new_data['train']['labels']) if lab_id not in label_pos]
            new_data['train']['text'] = [lab for lab_id,lab in enumerate(new_data['train']['text']) if lab_id not in label_pos]

            # print("len(new_data['train']['labels']): ",len(new_data['train']['labels']))
            # print("len(new_data['test']['labels']): ",len(new_data['test']['labels']))
        #TODO: ----------



        datasets = DatasetDict(
            {
                'train': Dataset.from_dict(new_data['train']),
                'test': Dataset.from_dict(new_data['test'])
            }
        )

    elif dataset_name == 'phone_sup':
        new_data = {}
        label2idx = {'+': 0, '-': 1}

        new_data['train'] = {}
        new_data['train']['text'] = []
        new_data['train']['labels'] = []
        for ds in ['train', 'dev', 'test']:
            with open('./data/absa/dat/Bing5Domains/Nokia6610/{}.json'.format(ds), 'r') as f:
                tmp_data = json.load(f)
            for dt in tmp_data:
                new_data['train']['labels'].append(label2idx[tmp_data[dt]['polarity']])
                new_data['train']['text'].append(tmp_data[dt]['term'] + ' ' + tokenizer.sep_token + tmp_data[dt]['sentence'])

        new_data['test'] = {}
        new_data['test']['text'] = []
        new_data['test']['labels'] = []
        for ds in ['train', 'dev', 'test']:
            with open('./data/absa/dat/Bing9Domains/Nokia6600/{}.json'.format(ds), 'r') as f:
                tmp_data = json.load(f)
            for dt in tmp_data:
                new_data['test']['labels'].append(label2idx[tmp_data[dt]['polarity']])
                new_data['test']['text'].append(tmp_data[dt]['term'] + ' ' + tokenizer.sep_token + tmp_data[dt]['sentence'])

        #TODO: ----------

        # we may re-partitial, by classes
        train_ratio = 0.7
        num_label = len(label2idx)
        total_num = len(new_data['train']['labels'])
        print("total_num: ", total_num)
        print("len(new_data['test']['labels']): ", len(new_data['test']['labels']))

        for label in range(num_label):
            num_takeout = int((total_num * (1-train_ratio)) // num_label)
            label_pos = [lab_id for lab_id,lab in enumerate(new_data['train']['labels']) if lab == label][:num_takeout]
            # print('num_takeout: ',num_takeout)
            label_takeout = [lab for lab_id,lab in enumerate(new_data['train']['labels']) if lab_id in label_pos]
            text_takeout = [lab for lab_id,lab in enumerate(new_data['train']['text']) if lab_id in label_pos]
            # print('label_takeout: ',len(label_takeout))

            new_data['test']['labels'] += label_takeout
            new_data['test']['text'] += text_takeout

            new_data['train']['labels'] = [lab for lab_id,lab in enumerate(new_data['train']['labels']) if lab_id not in label_pos]
            new_data['train']['text'] = [lab for lab_id,lab in enumerate(new_data['train']['text']) if lab_id not in label_pos]

            # print("len(new_data['train']['labels']): ",len(new_data['train']['labels']))
            # print("len(new_data['test']['labels']): ",len(new_data['test']['labels']))


        datasets = DatasetDict(
            {
                'train': Dataset.from_dict(new_data['train']),
                'test': Dataset.from_dict(new_data['test'])
            }
        )


    elif dataset_name not in implemented_datasets:
        raise NotImplementedError

    return datasets