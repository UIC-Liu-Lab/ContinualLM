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
from networks.baselines import ewc, hat, softmask, memory


def compute(self,model, train_loader_subset, self_fisher,mask_pre, buffer,accelerator):

    accelerator.wait_for_everyone()
    config = accelerator.unwrap_model(model).model.config

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.model.save_pretrained(self.args.output_dir)
        self.args.tokenizer.save_pretrained(self.args.output_dir)
        if 'adapter' in self.args.baseline:
            unwrapped_model.model.save_adapter(self.args.output_dir, 'adapter')
        if 'prompt' in self.args.baseline or 'l2p' in self.args.baseline:
            torch.save(unwrapped_model.model.keys, os.path.join(self.args.output_dir,  'keys'))
            torch.save(unwrapped_model.model.prompt_pool, os.path.join(self.args.output_dir, 'prompt_pool'))

    if 'ewc' in self.args.baseline:
        ewc.compute(train_loader_subset,model, self_fisher, accelerator, self.args)
    elif 'adapter_hat' in self.args.baseline \
            or 'transformer_hat' in self.args.baseline \
            or 'adapter_bcl' in self.args.baseline\
            or 'adapter_classic' in self.args.baseline:
        self.args.s = self.args.smax
        mask = self.mask(model,accelerator,self.args)
        hat.compute(model, accelerator, mask_pre, mask, self.get_view_for, self.args)
    elif 'derpp' in self.args.baseline:
        # Add data to the buffer
        if accelerator.is_main_process:
            buffer.add_from_loader(model, train_loader_subset)
            buffer_path = os.path.join(self.args.output_dir + '../', 'buffer')
            buffer.save(buffer_path)
            print('current buffer size: ', buffer.get_size())
            print('total buffer size: ', buffer.buffer_size)

    if self.args.softmask_compute is not None:
        if 'after_mlm' in self.args.softmask_compute:
            softmask.compute_impt(args=self.args, config=config, model=model,
                                           eval_dataloader=train_loader_subset, accelerator=accelerator,
                                           prune_loss='after_mlm')

    return self

