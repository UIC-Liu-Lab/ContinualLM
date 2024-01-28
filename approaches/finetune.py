
import logging
import math
import os
import torch
from tqdm.auto import tqdm
from networks import prompt
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
import numpy as np

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter
from utils import utils

class Appr(object):

    def __init__(self,args):
        super().__init__()
        self.args=args
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        return


    # TODO: Multiple-GPU supprt

    def train(self,model,accelerator,train_loader, test_loader):

        # Set the optimizer
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr,
                          weight_decay=self.args.weight_decay)

        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.epoch * num_update_steps_per_epoch
        else:
            self.args.epoch = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )

        # Prepare everything with the accelerator
        model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

        logger.info("***** Running training *****")
        logger.info( f"Pretrained Model = {self.args.model_name_or_path},  Dataset name = {self.args.dataset_name}, seed = {self.args.seed}")

        summary_path = f'{self.args.output_dir}../{self.args.dataset_name}_finetune_summary'
        print(f'summary_path: {summary_path}')

        for epoch in range(self.args.epoch):
            print("Epoch {} started".format(epoch))
            train_acc, training_loss = self.train_epoch(model, optimizer, train_loader, accelerator, lr_scheduler)
            print("train acc = {:.4f}, training loss = {:.4f}".format(train_acc, training_loss))

        micro_f1, macro_f1, acc, test_loss = self.eval(model, test_loader, accelerator)

        if self.args.dataset_name in ['chemprot_sup', 'rct_sample_sup']:
            macro_f1 = micro_f1  # we report micro instead

        logger.info(
            "{} On {}, last epoch macro_f1 = {:.4f}, acc = {:.4f} (seed={})".format(self.args.model_name_or_path,
                                                                                    self.args.dataset_name, macro_f1,
                                                                                    acc, self.args.seed))

        if accelerator.is_main_process:

            progressive_f1_path = f'{self.args.output_dir}/../progressive_f1_{self.args.seed}'
            progressive_acc_path = f'{self.args.output_dir}/../progressive_acc_{self.args.seed}'

            print(f'Path of progressive f1 score: {progressive_f1_path}')
            print(f'Path of progressive accuracy: {progressive_acc_path}')

            if os.path.exists(progressive_f1_path):
                f1s = np.loadtxt(progressive_f1_path)
                accs = np.loadtxt(progressive_acc_path)

            else:
                f1s = np.zeros((self.args.ntasks, self.args.ntasks), dtype=np.float32)
                accs = np.zeros((self.args.ntasks, self.args.ntasks), dtype=np.float32)

            f1s[self.args.pt_task][self.args.ft_task] = macro_f1
            np.savetxt(progressive_f1_path, f1s, '%.4f', delimiter='\t')

            accs[self.args.pt_task][self.args.ft_task] = acc
            np.savetxt(progressive_acc_path, accs, '%.4f', delimiter='\t')

            if self.args.ft_task == self.args.ntasks - 1:  # last ft task, we need a final one
                final_f1 = f'{self.args.output_dir}/../f1_{self.args.seed}'
                final_acc = f'{self.args.output_dir}/../acc_{self.args.seed}'

                forward_f1 = f'{self.args.output_dir}/../forward_f1_{self.args.seed}'
                forward_acc = f'{self.args.output_dir}/../forward_acc_{self.args.seed}'

                print(f'Final f1 score: {final_f1}')
                print(f'Final accuracy: {final_acc}')

                if self.args.baseline == 'one':
                    with open(final_acc, 'w') as file, open(final_f1, 'w') as f1_file:
                        for j in range(accs.shape[1]):
                            file.writelines(str(accs[j][j]) + '\n')
                            f1_file.writelines(str(f1s[j][j]) + '\n')

                else:
                    with open(final_acc, 'w') as file, open(final_f1, 'w') as f1_file:
                        for j in range(accs.shape[1]):
                            file.writelines(str(accs[-1][j]) + '\n')
                            f1_file.writelines(str(f1s[-1][j]) + '\n')

                    with open(forward_acc, 'w') as file, open(forward_f1, 'w') as f1_file:
                        for j in range(accs.shape[1]):
                            file.writelines(str(accs[j][j]) + '\n')
                            f1_file.writelines(str(f1s[j][j]) + '\n')


    def train_epoch(self,model, optimizer, dataloader, accelerator, lr_scheduler):
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
        model.train()
        train_acc = 0.0
        training_loss = 0.0
        total_num = 0.0
        for batch, inputs in enumerate(dataloader):
            if 'transformer_hat' in self.args.baseline:
                model_ori = accelerator.unwrap_model(model)
                head_importance, intermediate_importance, output_importance = model_ori.transformer_mask()
                res = model.model(**inputs, head_mask=head_importance, intermediate_mask=intermediate_importance,
                                output_mask=output_importance)

            else:
                res = model.model(**inputs)

            outp = res.logits
            loss = res.loss
            optimizer.zero_grad()
            accelerator.backward(loss)

            # for n,p in accelerator.unwrap_model(model).named_parameters():
            #     if p.grad is not None:
            #         print('n,p： ',n)

            optimizer.step()
            lr_scheduler.step()

            pred = outp.max(1)[1]

            predictions = accelerator.gather(pred)
            references = accelerator.gather(inputs['labels'])


            train_acc += (references == predictions).sum().item()
            training_loss += loss.item()
            total_num += references.size(0)

            progress_bar.update(1)
            # break
        return train_acc / total_num, training_loss / total_num

    def eval(self,model, dataloader, accelerator):
        model.eval()
        label_list = []
        prediction_list = []
        total_loss=0
        total_num=0
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
        with torch.no_grad():
            for batch, inputs in enumerate(dataloader):
                input_ids = inputs['input_ids']

                res = model.model(**inputs, return_dict=True)

                real_b=input_ids.size(0)
                loss = res.loss
                outp = res.logits
                if self.args.problem_type != 'multi_label_classification':
                    pred = outp.max(1)[1]
                else:
                    pred = outp.sigmoid() > 0.5

                total_loss+=loss.data.cpu().numpy().item()*real_b
                total_num+=real_b

                predictions = accelerator.gather(pred)
                references = accelerator.gather(inputs['labels'])

                label_list += references.cpu().numpy().tolist() # we may use multi-node
                prediction_list += predictions.cpu().numpy().tolist()
                progress_bar.update(1)
                # break

        micro_f1 = f1_score(label_list, prediction_list, average='micro')
        macro_f1 = f1_score(label_list, prediction_list, average='macro')
        accuracy = sum([float(label_list[i] == prediction_list[i]) for i in range(len(label_list))]) * 1.0 / len(prediction_list)


        return micro_f1, macro_f1, accuracy,total_loss/total_num

