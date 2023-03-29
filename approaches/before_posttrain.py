import logging
import math

import numpy as np
import os
import torch
from tqdm.auto import tqdm
from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
    Adafactor
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
from utils import utils
from networks.baselines import ewc, hat, softmask, memory, demix


        # before training ***********************************************************************************************
def prepare(self,model, train_loader_subset, train_loader_subset_dataset, accelerator):
    self_fisher = None
    mask_pre = None
    mask_back = None
    buffer = None
    head_impt = None
    intermediate_impt = None
    output_impt = None


    if 'ewc' in self.args.baseline:
        if os.path.exists(os.path.join(self.args.output_dir + '../', 'fisher')):
            print('load fisher matrix **************')
            self_fisher = torch.load(os.path.join(self.args.output_dir + '../', 'fisher'))
            for k, v in self_fisher.items():
                self_fisher[k] = self_fisher[k].cuda()

    elif 'adapter_hat' in self.args.baseline \
            or 'transformer_hat' in self.args.baseline \
            or 'adapter_bcl' in self.args.baseline \
            or 'adapter_classic' in self.args.baseline:  # BCL included HAT
        print('load mask matrix **************')
        if os.path.exists(os.path.join(self.args.output_dir + '../', 'mask_pre')):
            mask_pre = torch.load(os.path.join(self.args.output_dir + '../', 'mask_pre'))
            mask_back = torch.load(os.path.join(self.args.output_dir + '../', 'mask_back'))

            for k, v in mask_pre.items():
                mask_pre[k] = mask_pre[k].cuda()

            for k, v in mask_back.items():
                mask_back[k] = mask_back[k].cuda()

    elif 'derpp' in self.args.baseline:
        buffer = memory.Buffer(int(self.args.replay_sample_per_task * self.args.ntasks),args=self.args)
        if self.args.pt_task > 0:
            buffer.load(os.path.join(self.args.prev_output, 'buffer'))

    elif self.args.pt_task > 0 and 'adapter_demix' in self.args.baseline:  # initialize the new adapter using the nearest adapter
        model = demix.compute(train_loader_subset, train_loader_subset_dataset, model, accelerator,self.args)

    if 'dga' in self.args.baseline or 'das' in self.args.baseline:
        train_loader_prune = accelerator.prepare(train_loader_subset)
        config = accelerator.unwrap_model(model).model.config

        if 'before_distill' in self.args.softmask_compute and (self.args.pt_task == 0 or 'dga' in self.args.baseline): # one and dga are the same

            config = accelerator.unwrap_model(model).model.config
            softmask.compute_impt(args=self.args, config=config, model=model,
                                                 eval_dataloader=train_loader_prune, accelerator=accelerator,
                                                 prune_loss='before_distill')

        if 'before_mlm' in self.args.softmask_compute and self.args.pt_task == 0:  # only for wiki in task 0

            model = accelerator.prepare(model)
            softmask.compute_impt(args=self.args, config=config, model=model,
                                                 eval_dataloader=train_loader_prune, accelerator=accelerator,
                                                 prune_loss='before_mlm')

        accelerator.wait_for_everyone()
        head_impt, intermediate_impt, output_impt = softmask.accumulate_impt(self.args)

        if accelerator.is_main_process:
            print('head_impt: ', head_impt)
            print('intermediate_impt: ', intermediate_impt)
            print('output_impt: ', output_impt)


    if 'head_mask' in self.args.layer_to_mask:
        head_impt = head_impt
    if 'intermediate_mask' in self.args.layer_to_mask:
        intermediate_impt = intermediate_impt
    if 'output_mask' in self.args.layer_to_mask:
        output_impt = output_impt

    return self,model,head_impt, intermediate_impt, output_impt,self_fisher,mask_pre,mask_back,buffer

