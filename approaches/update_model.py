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

def update(self,model,optimizer,outputs,loss,writer,lr_scheduler,progress_bar,global_step,completed_steps,accelerator):
    optimizer.step()
    lr_scheduler.step()

    optimizer.zero_grad()
    progress_bar.update(1)
    completed_steps += 1

    if 'adapter_hat' in self.args.baseline \
            or 'transformer_hat' in self.args.baseline \
            or 'adapter_bcl' in self.args.baseline \
            or 'adapter_classic' in self.args.baseline:
        # Constrain embeddings
        for n, p in model.named_parameters():
            if 'adapters.e' in n or 'model.e' in n:
                p.data = torch.clamp(p.data, -self.args.thres_emb, self.args.thres_emb)
    progress_bar.set_description(
        'Train Iter (loss=%5.3f)' % loss.item())  # show the loss, mean while

    # Set up logging ------------
    if accelerator.is_main_process:
        utils.model.log_loss(writer, scalar_value=loss.item(), global_step=global_step)
        utils.model.log_loss(writer, loss_name=' MLM loss', scalar_value=outputs.loss.item(),
                             global_step=global_step)
        if 'dga' in self.args.baseline or 'das' in self.args.baseline:
            utils.model.log_loss(writer, loss_name=' contrastive loss',
                                 scalar_value=outputs.contrast_loss.item(), global_step=global_step)
        if 'distill' in self.args.baseline:
            utils.log_loss(writer, loss_name=' distill loss',
                           scalar_value=outputs.distill_loss.item(),
                           global_step=global_step)
        if 'simcse' in self.args.baseline:
            utils.log_loss(writer, loss_name=' simcse loss',
                           scalar_value=outputs.simcse_loss.item(), global_step=global_step)
        if 'tacl' in self.args.baseline:
            utils.log_loss(writer, loss_name=' tacl loss',
                           scalar_value=outputs.tacl_loss.item(), global_step=global_step)
        if 'taco' in self.args.baseline:
            utils.log_loss(writer, loss_name=' taco loss',
                           scalar_value=outputs.taco_loss.item(), global_step=global_step)
        if 'infoword' in self.args.baseline:
            utils.log_loss(writer, loss_name=' infoward loss',
                           scalar_value=outputs.infoword_loss.item(),
                           global_step=global_step)
    return