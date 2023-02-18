import logging
import math

from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
    Adafactor
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
import nltk

def postprocess_text(preds, labels):

    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def get_warmup_steps(num_training_steps,args):
    """
    Get number of steps used for a linear warmup.
    """
    warmup_steps = args.num_warmup_steps if args.num_warmup_steps > 0 else math.ceil(num_training_steps * args.warmup_ratio)
    return warmup_steps



def lookfor_optimize(model,args):

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            'name': [n for n, p in model.named_parameters()
                     if p.requires_grad and not any(nd in n for nd in no_decay)],
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate
        },
        {
            'name': [n for n, p in model.named_parameters()
                     if p.requires_grad and any(nd in n for nd in no_decay)],
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.learning_rate

        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters)

    return optimizer


