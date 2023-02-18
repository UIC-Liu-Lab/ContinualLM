


from tqdm.auto import tqdm
import torch
import torch.distributed as dist
import os
import math
import numpy as np


def compute(eval_dataloader,eval_dataset,model,accelerator,args):


    cur_task = args.task

    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)


    model.eval()

    perplexties = []
    for candidate_t in range(cur_task):
        progress_bar = tqdm(range(len(eval_dataloader)))
        losses = []
        args.task = candidate_t
        for step, batch in enumerate(eval_dataloader): # dataloader can be repeated
            bsz = batch['input_ids'].size(0)
            with torch.no_grad():
                outputs = model(batch)
            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(bsz)))
            progress_bar.update(1)
            progress_bar.set_description('DEMIX Iter (task=%1d)' % candidate_t)  # show the loss, mean while

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        perplexties.append(perplexity)


    demix_t = np.argmin(perplexties) # the smaller, the better

    #TODO: Not finish, need to do transfer in the adpater package
    raise NotImplementedError
    print('perplexties: ',perplexties)
    # print('demix_t: ',demix_t)
    for name, sub_module in model.named_modules():
        if isinstance(sub_module, MOE):
            sub_module.adapters[cur_task].transfer_weight(sub_module.adapters[demix_t])
    args.task = cur_task

    return model