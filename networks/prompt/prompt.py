import os
import pdb
import torch
import torch.nn as nn
from pathlib import Path
from torch.nn import functional as F


def initialize_prompt(
        model,
        n_tokens: int
) -> None:
    """
    Initialize the prompt_embed_pool (dict{task_it: nn.Embedding})
    """
    model.n_tokens = n_tokens  # Currently suppose n_tokens is the same for each task's prompt embedding.

    # Prompt embedding x is task-specific.
    # Embeddings for different tasks are maintained in a dict.
    model.prompt_embed_pool = nn.ModuleList()

    return model

def add_prompt_embedding(
        model,
        saved_embedding: nn.Embedding = None,
        random_range: float = 0.5
) -> None:
    """
    Add prompt embedding x into prompt_embed_pool for task t.
    """
    if saved_embedding is not None:
        if saved_embedding.weight.shape != (model.n_tokens, model.config.hidden_size):
            raise ValueError("The saved embedding has wrong shape.")
        model.prompt_embed_pool.append(saved_embedding)
        print('load prompt')

    else:
        init_prompt_value = torch.FloatTensor(model.n_tokens, model.config.hidden_size).uniform_(
            -random_range, random_range)
        embedding = nn.Embedding(model.n_tokens, model.config.hidden_size)
        embedding.weight = nn.parameter.Parameter(init_prompt_value)
        model.prompt_embed_pool.append(embedding)
        print('add prompt')
    return model

# all prompt put in the back ***********
def cat_learned_embedding_to_input(model, input_ids, t):
    """
    Concatenate the calculated prompt embedding and the original input embedding.
    Suppose MOE plugin is MOE(x, t), the calculated prompt embedding should be
    MOE(prompt_embed_pool[t], t)
    """
    # TODO: Pass in more parameters if MOE plugin gets more complicated.
    inputs_embeds = getattr(model, 'roberta').embeddings(input_ids)

    if len(list(inputs_embeds.shape)) == 2:
        inputs_embeds = inputs_embeds.unsqueeze(0)

    # [batch_size, n_tokens, n_embed]
    learned_embeds = model.prompt_embed_pool[t].weight.cuda().repeat(inputs_embeds.size(0), 1, 1)

    # X: n * e, P: p * e -> [P; X]: (p+n) * e
    inputs_embeds = torch.cat([inputs_embeds,learned_embeds], dim=1) # add at the end, I don't want to affect the classification token
    # TODO: in some cases, we may want to add the prompt to a specific position

    return inputs_embeds


def extend_labels(model, labels, ignore_index=-100) -> torch.Tensor:
    """
    Extend labels when training the language model using MLM to match
    the input_ids's shape.
    These pseudo labels will be ignored when calculating the MLM loss.
    This function shouldn't be call is we are using the model for end
    task (e.g. sequence classification)
    """
    if len(list(labels.shape)) == 1:
        labels = labels.unsqueeze(0)

    n_batches = labels.shape[0]
    return torch.cat(
        [
            labels,
            torch.full((n_batches, model.n_tokens), ignore_index).cuda(),
        ],
        dim=1,
    )


def extend_attention_mask(model, attention_mask):
    """
    Extend attention_mask to match the input_ids's shape.
    """
    if len(list(attention_mask.shape)) == 1:
        attention_mask = attention_mask.unsqueeze(0)

    n_batches = attention_mask.shape[0]
    return torch.cat(
        [attention_mask,torch.full((n_batches, model.n_tokens), 1).cuda()],
        dim=1,
    )