# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from typing import Tuple
from torchvision import transforms
import os
from tqdm.auto import tqdm
import torch.nn as nn
import json
import torch.distributed as dist

#TODO: MBPA++ https://github.com/h3lio5/episodic-lifelong-learning


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size: #total batch size
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size



#TODO: add logits


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device= 'cuda', n_tasks=None, args=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        self.args = args
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples','attention_mask','labels','logits','task']

    
        # assume it is fixed: 10/31/2022;


    def init_tensors(self,
                    examples: torch.Tensor,
                    attention_mask: torch.Tensor,
                     labels: torch.Tensor,
                     logits: torch.Tensor,
                     task: torch.Tensor) -> None:

        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,*attr.shape[1:]), dtype=typ, device=self.device))


    def get_size(self):
        return self.num_seen_examples


    def append_memeory_batch(self,batch):

        buf_inputs, buf_attention_mask, buf_labels, buf_logits, buf_task = self.get_data(batch['input_ids'].size(0))

        batch['input_ids'] = torch.cat([batch['input_ids'],buf_inputs])
        batch['attention_mask'] = torch.cat([batch['attention_mask'],buf_attention_mask])
        batch['labels'] = torch.cat([batch['labels'],buf_labels])
        batch['logits'] = torch.cat([batch['logits'],buf_logits])
        batch['task']  = torch.cat([batch['task'],buf_task])

        return batch


    def add_data(self, examples, attention_mask=None,  labels=None, logits=None, task=None):

        if not hasattr(self, 'examples'):
            self.init_tensors(examples, attention_mask, labels, logits, task)


        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if attention_mask is not None:
                    self.attention_mask[index] = attention_mask[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task is not None:
                    self.task[index] = task[i].to(self.device)


    def add_from_loader(self,model,train_dataloader):

        # Init
        progress_bar = tqdm(range(len(train_dataloader)))

        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            task = batch['task']
            outputs = model(batch)

            logits = outputs.hidden_states[-1]

            self.add_data(
                examples=self.gather_by_cat(input_ids),
                attention_mask=self.gather_by_cat(attention_mask),
                labels=self.gather_by_cat(labels),
                logits=self.gather_by_cat(logits),
                task=self.gather_by_cat(task))
            progress_bar.update(1)
            progress_bar.set_description('Memory Compute Iter ')  # show the loss, mean while

        return self

    def gather_by_cat(self,head_impt):
        head_impt_list = [torch.zeros_like(head_impt) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=head_impt_list,
                        tensor=head_impt.contiguous())  # everyone need to do this
        head_impt_cat = torch.cat(head_impt_list)
        return head_impt_cat

    def get_data(self, size: int, transform: transforms=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0




    def get_all_keys(self):
        # empty exclude
        outputs = self.key_encoder(self.examples[:self.num_seen_examples].long(),self.attention_mask[:self.num_seen_examples].long())

        return outputs.sequence_output[:, 0, :] # roberta/bert only

    def get_keys(self, input_ids, attention_mask):
        outputs = self.key_encoder(input_ids, attention_mask)
        return outputs.sequence_output[:, 0, :] # roberta/bert only



    def get_neighbours_til(self,keys):
        # if I know the task ID
        """
        Returns samples from buffer using nearest neighbour approach
        """

        samples = []

        for key in keys:

            indices = []

            for task_id, task in enumerate(self.task.cpu().numpy()[:self.num_seen_examples]):
                if task == self.args.eval_t:
                    indices.append(task_id)

            neighbours = (self.examples[indices],self.attention_mask[indices],self.labels[indices],self.logits[indices],self.task[indices])
            samples.append(neighbours)
            # print('self.task[indices]: ',self.task[indices])
        return samples


    def get_neighbours(self, keys, k=32):
        """
        Returns samples from buffer using nearest neighbour approach
        """
        all_keys = self.get_all_keys()

        # if self.args.task_name in self.args.classification:
        #     num_class = sum([_[1] for _ in self.args.taskcla[:self.args.ft_task]])
        #     k = k * num_class

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        samples = []
        for key in keys:
            sim = cos(key, all_keys)
            selection = torch.topk(sim, k)
            indices = selection.indices
            print('self.task[indices]: ',self.task[indices])
            print('self.task[indices]: ',len(self.task[indices]))

            neighbours = (self.examples[indices],self.attention_mask[indices],self.labels[indices],self.logits[indices],self.task[indices])
            samples.append(neighbours)

        return samples


    def process_lst(self,lst):
        return [np.array(x) for x in lst]

    def process_array(self,lst):
        return [x.tolist() for x in lst]

    def process_int(self,lst):
        return [int(x) for x in lst]

    def save(self, path):
        obj = [self.examples, self.attention_mask, self.labels, self.logits, self.task, self.num_seen_examples]
        torch.save(obj, path)

    def load(self, path):
        self.examples, self.attention_mask, self.labels, self.logits, self.task, self.num_seen_examples = torch.load(path,map_location='cpu')

        self.examples = self.examples.long().cuda()
        self.attention_mask = self.attention_mask.long().cuda()
        self.labels = self.labels.long().cuda()
        self.logits = self.logits.long().cuda()
        self.task = self.task.long().cuda()