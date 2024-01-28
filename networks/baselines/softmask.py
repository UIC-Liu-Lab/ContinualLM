from tqdm.auto import tqdm
import torch
import math
import numpy as np
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
from torch import nn
from itertools import zip_longest
import utils
import os
import torch.distributed as dist
import torch.autograd as autograd
from os import path


def gather_by_mean(head_impt):
    head_impt_list = [torch.zeros_like(head_impt) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list=head_impt_list,
                    tensor=head_impt.contiguous())
    head_impt_list = torch.stack(head_impt_list)
    head_impt = torch.mean(head_impt_list, dim=0)
    return head_impt


def impt_norm(impt):
    tanh = torch.nn.Tanh()
    for layer in range(impt.size(0)):
        impt[layer] = (impt[layer] - impt[layer].mean()) / impt[
            layer].std()  # 2D, we need to deal with this for each layer
    impt = tanh(impt).abs()

    return impt


def initial_impt(config):


    n_encoder_layer, n_encoder_heads = config.num_hidden_layers, config.num_attention_heads

    intermediate_impt = torch.zeros(n_encoder_layer, config.intermediate_size).cuda()
    intermediate_mask = torch.ones(n_encoder_layer, config.intermediate_size).cuda()
    intermediate_mask.requires_grad_(requires_grad=True)

    output_impt = torch.zeros(n_encoder_layer, config.hidden_size).cuda()
    output_mask = torch.ones(n_encoder_layer, config.hidden_size).cuda()
    output_mask.requires_grad_(requires_grad=True)

    head_impt = torch.zeros(n_encoder_layer, n_encoder_heads).cuda()
    head_mask = torch.ones(n_encoder_layer, n_encoder_heads).cuda()
    head_mask.requires_grad_(requires_grad=True)


    tot_tokens = 0.0

    return  head_impt, intermediate_impt, output_impt,head_mask, intermediate_mask, output_mask, tot_tokens



def compute_impt(args,config, model,eval_dataloader,accelerator,prune_loss=None):
    # model.train() # Train mode results in NAN

    # MLM/Distill loss *****************************
    head_impt, intermediate_impt, output_impt, \
    head_mask, intermediate_mask, output_mask, tot_tokens = initial_impt(config)


    for step, inputs in enumerate(tqdm(eval_dataloader, desc=f"Iteration {prune_loss}")):
        outputs = model(inputs,
                        head_mask=head_mask,intermediate_mask=intermediate_mask,output_mask=output_mask,
                        prune_loss=prune_loss)
        loss = outputs.loss

        # One can also only deal with the auto grad
        # g, = autograd.grad(loss, head_mask)
        # head_impt+= g.squeeze().detach()

        accelerator.backward(loss)

        head_impt += head_mask.grad.detach()
        intermediate_impt += intermediate_mask.grad.detach()
        output_impt += output_mask.grad.detach()

        tot_tokens += inputs["attention_mask"].float().detach().sum().data



    # Normalize
    head_impt /= tot_tokens

    intermediate_impt /= tot_tokens
    output_impt /= tot_tokens

    accelerator.wait_for_everyone()

    head_impt = gather_by_mean(head_impt)
    intermediate_impt = gather_by_mean(intermediate_impt)
    output_impt = gather_by_mean(output_impt)

    if accelerator.is_main_process:
        np.save(f'{args.output_dir}/{prune_loss}{args.pt_task}/head_impt.npy', head_impt.detach().cpu().numpy())
        np.save(f'{args.output_dir}/{prune_loss}{args.pt_task}/intermediate_impt.npy',intermediate_impt.detach().cpu().numpy())
        np.save(f'{args.output_dir}/{prune_loss}{args.pt_task}/output_impt.npy', output_impt.detach().cpu().numpy())


    return head_impt, intermediate_impt, output_impt



def accumulate_impt(args):
    head_impt_list = []
    intermediate_impt_list = []
    output_impt_list = []

    for impt_dir_id, impt_dir in enumerate(args.saved_output_dir):
        print(f'Read importance from {impt_dir}')

        head_impt_path = f'{impt_dir}/head_impt.npy'
        intermediate_impt_path = f'{impt_dir}/intermediate_impt.npy'
        output_impt_path = f'{impt_dir}/output_impt.npy'

        if not path.exists(head_impt_path):
            print(f'Warning: file {head_impt_path} does not exist')
            continue

        head_impt = torch.Tensor(np.load(head_impt_path)).cuda()
        head_impt = impt_norm(head_impt)
        head_impt_list.append(head_impt)

        intermediate_impt = torch.Tensor(np.load(intermediate_impt_path)).cuda()
        intermediate_impt = impt_norm(intermediate_impt)
        intermediate_impt_list.append(intermediate_impt)

        output_impt = torch.Tensor(np.load(output_impt_path)).cuda()
        output_impt = impt_norm(output_impt)
        output_impt_list.append(output_impt)

    if len(head_impt_list) > 0:
        head_impts = torch.stack(head_impt_list)
        head_impt, _ = head_impts.max(0)

        intermediate_impts = torch.stack(intermediate_impt_list)
        intermediate_impt, _ = intermediate_impts.max(0)

        output_impts = torch.stack(output_impt_list)
        output_impt, _ = output_impts.max(0)  # We take a max to accumulate

    else:
        head_impt, intermediate_impt, output_impt = None, None, None

    return head_impt, intermediate_impt, output_impt




def soft_mask_gradient(model, pre_head_impt, pre_intermediate_impt,pre_output_impt,accelerator, epoch,step,args):

    model_ori = accelerator.unwrap_model(model)

    if accelerator.is_main_process and pre_head_impt is not None and epoch < 1 and step < 1:
        if 'head_mask' in args.layer_to_mask:
            print(f'Head mask usage {(pre_head_impt.sum() / pre_head_impt.numel()).item()}')
        if 'intermediate_mask' in args.layer_to_mask:
            print(f'Intermediate mask usage {(pre_intermediate_impt.sum() / pre_intermediate_impt.numel()).item()}')
        if 'output_mask' in args.layer_to_mask:
            print(f'Output mask usage {(pre_output_impt.sum() / pre_output_impt.numel()).item()}')

    n_layers, n_heads = model_ori.model.config.num_hidden_layers, model_ori.model.config.num_attention_heads
    head_size = int(model_ori.model.config.hidden_size / model_ori.model.config.num_attention_heads)

    for layer in range(n_layers):

        if 'head_mask' in args.layer_to_mask:
            head_impt = pre_head_impt[layer].unsqueeze(-1).repeat((1, head_size))
            head_impt = head_impt.flatten()
            head_mask = 1 - head_impt

            model_ori.model.roberta.encoder.layer[layer].attention.self.query.weight.grad *= head_mask
            model_ori.model.roberta.encoder.layer[layer].attention.self.query.bias.grad *= head_mask

            model_ori.model.roberta.encoder.layer[layer].attention.self.key.weight.grad *= head_mask
            model_ori.model.roberta.encoder.layer[layer].attention.self.key.bias.grad *= head_mask

            model_ori.model.roberta.encoder.layer[layer].attention.self.value.weight.grad *= head_mask
            model_ori.model.roberta.encoder.layer[layer].attention.self.value.bias.grad *= head_mask

            model_ori.model.roberta.encoder.layer[layer].attention.output.dense.weight.grad *= head_mask
            model_ori.model.roberta.encoder.layer[layer].attention.output.dense.bias.grad *= head_mask

        if 'intermediate_mask' in args.layer_to_mask:
            intermediate_mask = (1 - pre_intermediate_impt[layer])
            model_ori.model.roberta.encoder.layer[
                layer].intermediate.dense.weight.grad *= intermediate_mask.unsqueeze(1)
            model_ori.model.roberta.encoder.layer[
                layer].intermediate.dense.bias.grad *= intermediate_mask

        if 'output_mask' in args.layer_to_mask:
            output_mask = (1 - pre_output_impt[layer])
            model_ori.model.roberta.encoder.layer[
                layer].output.dense.weight.grad *= output_mask.unsqueeze(1)
            model_ori.model.roberta.encoder.layer[layer].output.dense.bias.grad *= output_mask

