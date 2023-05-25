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
from copy import deepcopy

        # before training ***********************************************************************************************
def compute(self,model,batch,head_impt,intermediate_impt,output_impt,self_fisher,mask_pre,train_loader,step,accelerator):

    self.args.s = (self.args.smax - 1 / self.args.smax) * step / len(train_loader) + 1 / self.args.smax # for HAT based model

    # learns outputs ------------
    if 'ewc' in self.args.baseline:
        outputs = model(batch, self_fisher=self_fisher)
    elif 'adapter_hat' in self.args.baseline \
            or 'adapter_bcl' in self.args.baseline \
            or 'adapter_classic' in self.args.baseline:
        masks = self.mask(model, accelerator, self.args)
        outputs = model(batch, masks=masks, mask_pre=mask_pre)
    elif 'transformer_hat' in self.args.baseline:
        model_ori = accelerator.unwrap_model(model)
        head_importance, intermediate_importance, output_importance = model_ori.model.transformer_mask()
        masks = self.mask(model, accelerator, self.args)  # need mask
        outputs = model(batch, head_mask=head_importance,
                        intermediate_mask=intermediate_importance, output_mask=output_importance,
                        masks=masks, mask_pre=mask_pre)

    elif 'dga' in self.args.baseline or 'das' in self.args.baseline:
        outputs = model(batch,
                        head_mask=head_impt,
                        intermediate_mask=intermediate_impt,
                        output_mask=output_impt)
    else:
        outputs = model(batch)
    return self, model, outputs