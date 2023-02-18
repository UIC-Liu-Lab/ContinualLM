import copy
import shutil
import argparse
import logging
import math
import os
import random
import sys
import torch
import datasets
import transformers
from accelerate import Accelerator, DistributedType
from tqdm.auto import tqdm
from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoTokenizer,
    AutoConfig,
    RobertaTokenizer,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
    SchedulerType,
    set_seed,
)

from utils import utils
from approaches import after_posttrain, before_posttrain, compute_loss, compute_gradient, update_model
from networks.baselines import ewc, hat, softmask, memory


logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class Appr(object):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        return


    def train(self, model, accelerator, train_dataset,train_loader, train_loader_subset,train_loader_subset_dataset):


        optimizer = utils.optimize.lookfor_optimize(model,self.args)

        # Prepare everything with our `accelerator`.
        model, optimizer, train_loader, train_loader_subset = accelerator.prepare(model, optimizer, train_loader, train_loader_subset)

        # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
        if accelerator.distributed_type == DistributedType.TPU:
            model.tie_weights()

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)

        if self.args.max_samples is not None:
            self.args.max_train_steps = self.args.max_samples // (
                    self.args.per_device_train_batch_size * accelerator.num_processes * self.args.gradient_accumulation_steps)

        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        else:
            self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        # Warm up can be important
        # warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1
        self.args.num_warmup_steps = int(float(self.args.warmup_proportion) * float(self.args.max_train_steps))  # 0.1

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )

        # Train!
        total_batch_size = self.args.per_device_train_batch_size * accelerator.num_processes * self.args.gradient_accumulation_steps


        # before training ***********************************************************************************************

        self,model,head_impt, intermediate_impt, output_impt,self_fisher,mask_pre,mask_back,buffer \
            = before_posttrain.prepare(self,model, train_loader_subset,train_loader_subset_dataset, accelerator)

        # before training ***********************************************************************************************

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
        self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch


        if accelerator.is_main_process:
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {len(train_dataset)}")
            logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
            logger.info(f"  Total samples = {self.args.max_train_steps * total_batch_size}")
            logger.info(
                f"  Learning Rate = {self.args.learning_rate}, Warmup Num = {self.args.num_warmup_steps}, Pre-trained Model = {self.args.model_name_or_path}")
            logger.info(
                f"  Seq ID = {self.args.idrandom}, Task id = {self.args.pt_task}, dataset name = {self.args.dataset_name}")
            logger.info(f"  Baseline = {self.args.baseline}, Smax = {self.args.smax}")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        global_step = 0  # This will be used by CLMOE if we choose 'auto_encoder' as the route type.

        writer = None
        if accelerator.is_main_process:
            tensorboard_file = os.path.join(self.args.output_dir, str(self.args.dataset_name) + '_log')
            print('tensorboard_file: ', tensorboard_file)
            if os.path.isdir(tensorboard_file):
                shutil.rmtree(tensorboard_file)
            writer = utils.model.setup_writer(tensorboard_file)

        try:
            if not self.args.eval_only:
                for epoch in range(self.args.num_train_epochs):
                    # break
                    model.train()
                    for step, batch in enumerate(train_loader):

                        self, model, outputs = compute_loss.compute(self,model,batch,head_impt,intermediate_impt,output_impt,self_fisher,mask_pre,accelerator)
                        loss = outputs.loss  # loss 1
                        model = compute_gradient.compute(self,model,head_impt, intermediate_impt, output_impt,batch, loss,buffer,mask_back,outputs,epoch,step,accelerator)
                        global_step += 1

                        if step % self.args.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                            update_model.update(self,model,optimizer,outputs,loss,writer,lr_scheduler,progress_bar,global_step,completed_steps,accelerator)

                        # break
                        if completed_steps >= self.args.max_train_steps:
                            break

        except KeyboardInterrupt:  # even if contro-C, I still want to save model
            return
 

        after_posttrain.compute(self, model, train_loader_subset, self_fisher,mask_pre, buffer,accelerator)
