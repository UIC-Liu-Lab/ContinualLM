
import torch


########################################################################################################################

def lookfor_baseline_variable(self, args):

    self.tsv_para = \
        ['module.model.roberta.' + coder + '.layer.' + str(
            layer_id) + '.output.adapters.adapter.adapter_down.' + adapter + '.adapters.capsule_net.tsv_capsules.route_weights'
         for layer_id in range(12) for coder in ['encoder'] for adapter in ['linear_down', 'linear_up']] + \
        ['module.model.roberta.encoder.layer.' + str(
            layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.route_weights'
         for layer_id in range(12) for coder in ['encoder'] for adapter in ['linear_down', 'linear_up']] + \
        ['module.model.roberta.encoder.layer.' + str(
            layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.semantic_capsules.fc1.' + str(
            c_t) + '.weight'
         for c_t in range(args.ntasks) for layer_id in range(12) for coder in ['encoder'] for adapter in
         ['linear_down', 'linear_up']] + \
        ['module.model.roberta.encoder.layer.' + str(
            layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.semantic_capsules.fc1.' + str(
            c_t) + '.bias'
         for c_t in range(args.ntasks) for layer_id in range(12) for coder in ['encoder'] for adapter in
         ['linear_down', 'linear_up']] + \
        ['module.model.roberta.encoder.layer.' + str(
            layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.semantic_capsules.fc2.' + str(
            c_t) + '.weight'
         for c_t in range(args.ntasks) for layer_id in range(12) for coder in ['encoder'] for adapter in
         ['linear_down', 'linear_up']] + \
        ['module.model.roberta.encoder.layer.' + str(
            layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.semantic_capsules.fc2.' + str(
            c_t) + '.bias'
         for c_t in range(args.ntasks) for layer_id in range(12) for coder in ['encoder'] for adapter in
         ['linear_down', 'linear_up']] + \
        ['module.model.roberta.encoder.layer.' + str(
            layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.fc1.' + str(
            c_t) + '.weight'
         for c_t in range(args.ntasks) for layer_id in range(12) for coder in ['encoder'] for adapter in
         ['linear_down', 'linear_up']] + \
        ['module.model.roberta.encoder.layer.' + str(
            layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.fc1.' + str(
            c_t) + '.bias'
         for c_t in range(args.ntasks) for layer_id in range(12) for coder in ['encoder'] for adapter in
         ['linear_down', 'linear_up']] + \
        ['module.model.roberta.encoder.layer.' + str(
            layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.fc2.' + str(
            c_t) + '.weight'
         for c_t in range(args.ntasks) for layer_id in range(12) for coder in ['encoder'] for adapter in
         ['linear_down', 'linear_up']] + \
        ['module.model.roberta.encoder.layer.' + str(
            layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.fc2.' + str(
            c_t) + '.bias'
         for c_t in range(args.ntasks) for layer_id in range(12) for coder in ['encoder'] for adapter in
         ['linear_down', 'linear_up']]

    return self


def get_view_for_tsv(n, model_ori, args):
    # some weights should not affect eacher other, even if they are not covered by the fc mask

    t = args.ft_task

    for layer_id in range(12):
        if n == 'module.model.roberta.encoder.layer.' + str(
                layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.route_weights':
            return model_ori.model.roberta.encoder.layer[
                layer_id].output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[
                t].data.view(1, -1, 1, 1)


        if n == 'module.model.roberta.encoder.layer.' + str(
                layer_id) + '.output.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.route_weights':
            return model_ori.model.roberta.encoder.layer[
                layer_id].output.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[
                t].data.view(1, -1, 1, 1)


        for c_t in range(args.ntasks):
            if n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.semantic_capsules.fc1.' + str(
                c_t) + '.weight':
                return \
                    model_ori.model.roberta.encoder.layer[
                        layer_id].output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[
                        t][
                        c_t].data
            elif n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.semantic_capsules.fc1.' + str(
                c_t) + '.bias':
                return \
                    model_ori.model.roberta.encoder.layer[
                        layer_id].output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[
                        t][
                        c_t].data
            elif n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.semantic_capsules.fc2.' + str(
                c_t) + '.weight':
                return \
                    model_ori.model.roberta.encoder.layer[
                        layer_id].output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[
                        t][
                        c_t].data
            elif n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.semantic_capsules.fc2.' + str(
                c_t) + '.bias':
                return \
                    model_ori.model.roberta.encoder.layer[
                        layer_id].output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[
                        t][
                        c_t].data



            if n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.output.adapters.adapter.adapter_up.adapters.capsule_net.semantic_capsules.fc1.' + str(
                c_t) + '.weight':
                return \
                    model_ori.model.roberta.encoder.layer[
                        layer_id].output.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data
            elif n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.output.adapters.adapter.adapter_up.adapters.capsule_net.semantic_capsules.fc1.' + str(
                c_t) + '.bias':
                return \
                    model_ori.model.roberta.encoder.layer[
                        layer_id].output.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data
            elif n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.output.adapters.adapter.adapter_up.adapters.capsule_net.semantic_capsules.fc2.' + str(
                c_t) + '.weight':
                return \
                    model_ori.model.roberta.encoder.layer[
                        layer_id].output.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data
            elif n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.output.adapters.adapter.adapter_up.adapters.capsule_net.semantic_capsules.fc2.' + str(
                c_t) + '.bias':
                return \
                    model_ori.model.roberta.encoder.layer[
                        layer_id].output.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data


            for m_t in range(3):
                if n == 'module.model.roberta.encoder.layer.' + str(
                        layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.transfer_capsules.convs3.' + str(
                    c_t) + '.' + str(m_t) + '.weight':
                    return model_ori.model.roberta.encoder.layer[
                        layer_id].output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[
                        t][c_t].data
                if n == 'module.model.roberta.encoder.layer.' + str(
                        layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.transfer_capsules.convs3.' + str(
                    c_t) + '.' + str(m_t) + '.bias':
                    return model_ori.model.roberta.encoder.layer[
                        layer_id].output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[
                        t][c_t].data
                if n == 'module.model.roberta.encoder.layer.' + str(
                        layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.transfer_capsules.convs2.' + str(
                    c_t) + '.' + str(m_t) + '.weight':
                    return model_ori.model.roberta.encoder.layer[
                        layer_id].output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[
                        t][c_t].data
                if n == 'module.model.roberta.encoder.layer.' + str(
                        layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.transfer_capsules.convs2.' + str(
                    c_t) + '.' + str(m_t) + '.bias':
                    return model_ori.model.roberta.encoder.layer[
                        layer_id].output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[
                        t][c_t].data
                if n == 'module.model.roberta.encoder.layer.' + str(
                        layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.transfer_capsules.convs1.' + str(
                    c_t) + '.' + str(m_t) + '.weight':
                    return model_ori.model.roberta.encoder.layer[
                        layer_id].output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[
                        t][c_t].data
                if n == 'module.model.roberta.encoder.layer.' + str(
                        layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.transfer_capsules.convs1.' + str(
                    c_t) + '.' + str(m_t) + '.bias':
                    return model_ori.model.roberta.encoder.layer[
                        layer_id].output.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[
                        t][c_t].data

                if n == 'module.model.roberta.encoder.layer.' + str(
                        layer_id) + '.output.adapters.adapter.adapter_up.adapters.capsule_net.transfer_capsules.convs3.' + str(
                    c_t) + '.' + str(m_t) + '.weight':
                    return model_ori.model.roberta.encoder.layer[
                        layer_id].output.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data
                if n == 'module.model.roberta.encoder.layer.' + str(
                        layer_id) + '.output.adapters.adapter.adapter_up.adapters.capsule_net.transfer_capsules.convs3.' + str(
                    c_t) + '.' + str(m_t) + '.bias':
                    return model_ori.model.roberta.encoder.layer[
                        layer_id].output.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data
                if n == 'module.model.roberta.encoder.layer.' + str(
                        layer_id) + '.output.adapters.adapter.adapter_up.adapters.capsule_net.transfer_capsules.convs2.' + str(
                    c_t) + '.' + str(m_t) + '.weight':
                    return model_ori.model.roberta.encoder.layer[
                        layer_id].output.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data
                if n == 'module.model.roberta.encoder.layer.' + str(
                        layer_id) + '.output.adapters.adapter.adapter_up.adapters.capsule_net.transfer_capsules.convs2.' + str(
                    c_t) + '.' + str(m_t) + '.bias':
                    return model_ori.model.roberta.encoder.layer[
                        layer_id].output.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data
                if n == 'module.model.roberta.encoder.layer.' + str(
                        layer_id) + '.output.adapters.adapter.adapter_up.adapters.capsule_net.transfer_capsules.convs1.' + str(
                    c_t) + '.' + str(m_t) + '.weight':
                    return model_ori.model.roberta.encoder.layer[
                        layer_id].output.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data
                if n == 'module.model.roberta.encoder.layer.' + str(
                        layer_id) + '.output.adapters.adapter.adapter_up.adapters.capsule_net.transfer_capsules.convs1.' + str(
                    c_t) + '.' + str(m_t) + '.bias':
                    return model_ori.model.roberta.encoder.layer[
                        layer_id].output.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data

    return 1  # if no condition is satified


def mask(model, accelerator, args):
    model_ori = accelerator.unwrap_model(model)

    masks = {}

    for layer_id in range(model_ori.config.num_hidden_layers):
        if 'adapter_hat' in args.baseline \
                or 'adapter_bcl' in args.baseline \
                or 'adapter_ctr' in args.baseline \
                or 'adapter_classic' in args.baseline:  # BCL included HAT

            fc1_key = 'module.model.roberta.encoder.layer.' + str(
                layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.fc1'  # gfc1
            fc2_key = 'module.model.roberta.encoder.layer.' + str(
                layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.fc2'  # gfc2

            masks[fc1_key], masks[fc2_key] = model_ori.model.roberta.encoder.layer[
                layer_id].output.adapters.adapter.adapter_down.linear_down.adapters.mask()

            fc1_key = 'module.model.roberta.encoder.layer.' + str(
                layer_id) + '.output.adapters.adapter.adapter_up.adapters.fc1'  # gfc1
            fc2_key = 'module.model.roberta.encoder.layer.' + str(
                layer_id) + '.output.adapters.adapter.adapter_up.adapters.fc2'  # gfc2

            masks[fc1_key], masks[fc2_key] = model_ori.model.roberta.encoder.layer[
                layer_id].output.adapters.adapter.adapter_up.adapters.mask()


    return masks


def get_view_for(n, p, masks, config, args):
    for layer_id in range(config.num_hidden_layers):

        if 'adapter_hat' in args.baseline \
                or 'adapter_bcl' in args.baseline \
                or 'adapter_ctr' in args.baseline \
                or 'adapter_classic' in args.baseline:  # BCL included HAT
            if n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.attention.output.adapters.adapter.adapter_down.linear_down.adapters.fc1.weight':
                return masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
            elif n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.attention.output.adapters.adapter.adapter_down.linear_down.adapters.fc1.bias':
                return masks[n.replace('.bias', '')].data.view(-1)
            elif n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.attention.output.adapters.adapter.adapter_down.linear_down.adapters.fc2.weight':
                post = masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
                pre = masks[n.replace('.weight', '').replace('fc2', 'fc1')].data.view(1, -1).expand_as(p)
                return torch.min(post, pre)
            elif n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.attention.output.adapters.adapter.adapter_down.linear_down.adapters.fc2.bias':
                return masks[n.replace('.bias', '')].data.view(-1)

            elif n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.fc1.weight':
                # print('not nont')
                return masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
            elif n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.fc1.bias':
                return masks[n.replace('.bias', '')].data.view(-1)
            elif n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.fc2.weight':
                post = masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
                pre = masks[n.replace('.weight', '').replace('fc2', 'fc1')].data.view(1, -1).expand_as(p)
                return torch.min(post, pre)
            elif n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.output.adapters.adapter.adapter_down.linear_down.adapters.fc2.bias':
                return masks[n.replace('.bias', '')].data.view(-1)


            if n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.attention.output.adapters.adapter.adapter_up.adapters.fc1.weight':
                return masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
            elif n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.attention.output.adapters.adapter.adapter_up.adapters.fc1.bias':
                return masks[n.replace('.bias', '')].data.view(-1)
            elif n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.attention.output.adapters.adapter.adapter_up.adapters.fc2.weight':
                post = masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
                pre = masks[n.replace('.weight', '').replace('fc2', 'fc1')].data.view(1, -1).expand_as(p)
                return torch.min(post, pre)
            elif n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.attention.output.adapters.adapter.adapter_up.adapters.fc2.bias':
                return masks[n.replace('.bias', '')].data.view(-1)

            elif n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.output.adapters.adapter.adapter_up.adapters.fc1.weight':
                # print('not nont')
                return masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
            elif n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.output.adapters.adapter.adapter_up.adapters.fc1.bias':
                return masks[n.replace('.bias', '')].data.view(-1)
            elif n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.output.adapters.adapter.adapter_up.adapters.fc2.weight':
                post = masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
                pre = masks[n.replace('.weight', '').replace('fc2', 'fc1')].data.view(1, -1).expand_as(p)
                return torch.min(post, pre)
            elif n == 'module.model.roberta.encoder.layer.' + str(
                    layer_id) + '.output.adapters.adapter.adapter_up.adapters.fc2.bias':
                return masks[n.replace('.bias', '')].data.view(-1)

    return None



