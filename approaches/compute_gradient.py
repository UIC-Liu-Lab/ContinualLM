import logging
import torch
from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
    Adafactor
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
import utils
from networks.baselines import ewc, hat, softmask, memory

def compute(self,model,head_impt, intermediate_impt, output_impt,batch, loss,buffer,mask_back,outputs,epoch,step,accelerator):

    # add loss ------------

    if 'derpp' in self.args.baseline \
            and not (buffer is None or buffer.is_empty()) \
            and step % self.args.replay_freq == 0:

        replay_batch = buffer.get_datadict(size=batch['input_ids'].shape[0])
        replay_outputs = model(replay_batch)

        loss += replay_outputs.loss * self.args.replay_beta
        loss += self.mse(
            replay_outputs.hidden_states[-1], replay_batch['logits']) * self.args.replay_alpha
        
    # We keep track of the loss at each epoch
    loss = loss / self.args.gradient_accumulation_steps
    
    if 'dga' in self.args.baseline or 'das' in self.args.baseline:
        contrast_loss = outputs.contrast_loss  # loss 1
        loss = loss + contrast_loss
    if 'distill' in self.args.baseline:
        distill_loss = outputs.distill_loss  # loss 1
        loss = loss + distill_loss
    if 'simcse' in self.args.baseline:
        simcse_loss = outputs.simcse_loss  # loss 1
        loss = loss + simcse_loss
    if 'tacl' in self.args.baseline:
        tacl_loss = outputs.tacl_loss  # loss 1
        loss = loss + tacl_loss
    if 'taco' in self.args.baseline:
        taco_loss = outputs.taco_loss  # loss 1
        loss = loss + taco_loss
    if 'infoword' in self.args.baseline:
        infoword_loss = outputs.infoword_loss  # loss 1
        loss = loss + infoword_loss
    # add loss ------------

    loss = loss / self.args.gradient_accumulation_steps
    # add model needs to be careful! make sure it is in parameters and please double check its gradient
    accelerator.backward(loss)  # sync

    if accelerator.is_main_process and epoch < 1 and step < 1:
        for n, p in accelerator.unwrap_model(model).named_parameters():
            if p.grad is not None:
                print('n,p： ', n, p.size())

    # modify the gradient ------------

    if self.args.pt_task > 0 and \
            ('adapter_hat' in self.args.baseline
             or 'transformer_hat' in self.args.baseline
             or 'adapter_bcl' in self.args.baseline
             or 'adapter_classic' in self.args.baseline):
        for n, p in model.named_parameters():
            if n in mask_back and p.grad is not None:
                p.grad.data *= mask_back[n]

    if 'adapter_hat' in self.args.baseline \
            or 'transformer_hat' in self.args.baseline \
            or 'adapter_bcl' in self.args.baseline \
            or 'adapter_classic' in self.args.baseline:
        # Compensate embedding gradients
        for n, p in model.named_parameters():
            if 'adapters.e' in n or ('model.e' in n and p.grad is not None):
                num = torch.cosh(torch.clamp(self.args.s * p.data, -self.args.thres_cosh,
                                             self.args.thres_cosh)) + 1
                den = torch.cosh(p.data) + 1
                p.grad.data *= self.args.smax / self.args.s * num / den

    # we need this even for the first task
    if 'dga' in self.args.baseline or 'das' in self.args.baseline:
        softmask.soft_mask_gradient(model, head_impt, intermediate_impt, output_impt, accelerator, epoch, step,
                                    self.args)

    # modify the gradient ------------

    return model