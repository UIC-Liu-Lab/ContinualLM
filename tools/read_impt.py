
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



def impt_norm(impt):
    tanh = torch.nn.Tanh()
    for layer in range(impt.size(0)):
        impt[layer] = (impt[layer] - impt[layer].mean()) / impt[
            layer].std()  # 2D, we need to deal with this for each layer
    impt = tanh(impt).abs()

    return impt


# List of five airlines to plot
# domains = ['base', 'restaurant_unsup_roberta', 'acl_unsup_roberta', 'ai_unsup_roberta','phone_unsup_roberta', 'pubmed_unsup_roberta', 'camera_unsup_roberta']


import seaborn as sns
import matplotlib.pyplot as plt
import os
# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)



with open('sequence_10', 'r') as f:
    datas = f.readlines()

for seq in ['0','1','2','3','4','5']:

    if seq in ['0']:
        base_dir = '/hdd_3/zke4/seq'+seq+'/640000samples/contrast_pre_as_general_proxy/'
        domains = ['base'] + [data+'_roberta' for data in datas[int(seq)].split()]
    if seq in ['1']:
        base_dir = '/sdb/zke4/seq'+seq+'/640000samples/contrast_pre_as_general_proxy/'
        domains = ['base'] + [data+'_roberta' for data in datas[int(seq)].split()]

    if seq in ['2']:
        base_dir = '/sdb/zke4/seq'+seq+'/640000samples/contrast_pre_as_general_proxy/'
        domains = ['base'] + [data+'_roberta' for data in datas[int(seq)].split()]

    if seq in ['3']:
        base_dir = '/hdd_3/zke4/seq'+seq+'/640000samples/contrast_pre_as_general_proxy/'
        domains = ['base'] + [data+'_roberta' for data in datas[int(seq)].split()]

    if seq in ['4']:
        base_dir = '/hdd_3/zke4/seq'+seq+'/640000samples/contrast_pre_as_general_proxy/'
        domains = ['base'] + [data+'_roberta' for data in datas[int(seq)].split()]


    if seq in ['5']:
        base_dir = '/hdd_3/zke4/seq'+seq+'/640000samples/contrast_pre_as_general_proxy/'
        domains = ['base','acl_unsup_roberta']


    os.makedirs('./impt/init', exist_ok=True)

    with open('./impt/read_impt_seq'+seq,'w') as impt_f:
        os.makedirs('./impt/seq' + seq, exist_ok=True)

        for domain_id, domain in enumerate(domains):

            print('domain: ',domain)

            # mean and std ----------------------
            impt = torch.Tensor(np.load(base_dir + domain + '/head_importance.npy'))
            head_impt = impt_norm(impt)
            mean_head = torch.mean(head_impt).item()
            std_head = torch.std(head_impt).item()

            print('mean_head: ',mean_head)
            print('std_head: ',std_head)

            impt = torch.Tensor(np.load(base_dir + domain + '/intermediate_importance.npy'))
            intermediate_impt = impt_norm(impt)
            mean_intermediate = torch.mean(intermediate_impt).item()
            std_intermediate = torch.std(intermediate_impt).item()

            print('mean_intermediate: ',mean_intermediate)
            print('std_intermediate: ',std_intermediate)

            impt = torch.Tensor(np.load(base_dir + domain + '/output_importance.npy'))
            output_impt = impt_norm(impt)
            mean_output = torch.mean(output_impt).item()
            std_output = torch.std(output_impt).item()

            print('mean_output: ',mean_output)
            print('std_output: ',std_output)

            impt_f.writelines(str(mean_head) + '\t' + str(std_head) + '\t' + str(mean_intermediate) + '\t' + str(std_intermediate) + '\t' + str(mean_output) + '\t' + str(std_output) + '\n')

            # mean and std -------------




            # Draw the density plot
            sns.set(style="darkgrid")
            # sns.histplot(data=df, x="impt", bins=5)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3,tight_layout=True)
            head_impt = head_impt.flatten()
            head_impt = head_impt.numpy()
            df = pd.DataFrame(head_impt, columns=['impt'])
            # ax1.set(style="darkgrid")
            sns.histplot(data=df, x="impt", bins=3, ax=ax1)

            if domain == 'base':
                ax1.set_title('head ' + domain.replace('base','init') + '_' + domains[1].replace('_unsup_roberta',''))
            else:
                ax1.set_title('head ' + domain.replace('_unsup_roberta',''))

            intermediate_impt = intermediate_impt.flatten()
            intermediate_impt = intermediate_impt.numpy()
            df = pd.DataFrame(intermediate_impt, columns=['impt'])
            sns.histplot(data=df, x="impt", bins=3, ax=ax2)

            if domain == 'base':
                ax2.set_title('intermediate ' + domain.replace('base','init') + '_' + domains[1].replace('_unsup_roberta',''))
            else:
                ax2.set_title('intermediate ' + domain.replace('_unsup_roberta',''))


            output_impt = output_impt.flatten()
            output_impt = output_impt.numpy()
            df = pd.DataFrame(output_impt, columns=['impt'])
            sns.histplot(data=df, x="impt", bins=3, ax=ax3)

            if domain == 'base':
                ax3.set_title('output ' + domain.replace('base','init') + '_' + domains[1].replace('_unsup_roberta',''))
            else:
                ax3.set_title('output ' + domain.replace('_unsup_roberta','') + ' task ' + str(domain_id))



            if domain == 'base':
                plt.savefig('./impt/seq' + seq + '/' + domain.replace('base','init') + '_' + domains[1].replace('_unsup_roberta','') + '.png')
                plt.savefig('./impt/init/' + domain.replace('base','init') + '_' + domains[1].replace('_unsup_roberta','') + '.png')

            else:
                plt.savefig('./impt/seq' + seq + '/' + domain.replace('_unsup_roberta','') + ' task ' + str(domain_id) + '.png')







            plt.cla()



exit()


#
# avg_cosine = np.zeros((6,12), dtype=np.float32)
#
# for layer_id in range(12):
#     for domain_id, domain in enumerate(domains):
#         # Subset to the airline
#         impt = torch.Tensor(
#             np.load("./" + domain + '_unsup_roberta/' + 'before_distill' + str(domain_id) + "/head_impt.npy"))
#         head_impt = impt_norm(impt)[layer_id]
#         cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
#
#         cosine = []
#         for other_domain_id, other_domain in enumerate(domains):
#             if other_domain != domain:
#                 other_impt = torch.Tensor(
#                     np.load("./" + other_domain + '_unsup_roberta/' + 'before_distill' + str(other_domain_id) + "/head_impt.npy"))
#                 other_head_impt = impt_norm(other_impt)[layer_id]
#                 cosine.append(cos(head_impt.view(1,-1), other_head_impt.view(1,-1)).item())
#
#         avg_cosine[domain_id][layer_id] = np.mean(cosine)
#
# np.savetxt('cosine',avg_cosine, '%.2f', delimiter='\t')


# # Iterate through the five airlines
for domain_id, domain in enumerate(domains):
    # Subset to the airline
    impt = torch.Tensor(np.load("./" + domain + '_unsup_roberta/' + 'before_distill'+str(domain_id)+"/head_impt.npy"))
    head_impt = impt_norm(impt)
    head_impt = head_impt.flatten()
    head_impt = head_impt.numpy()
    df = pd.DataFrame(head_impt, columns=['impt'])

    print('head_impt: ',head_impt)
    #cosine similarity between domain a and other doamin (average)

    # Draw the density plot
    sns.set(style="darkgrid")
    sns.histplot(data=df, x="impt",bins=5)

    # Plot formatting
    # plt.legend(prop={'size': 10}, title='Domains')
    plt.title('Domain ' + domain )
    # plt.title('Domain ' + domain + ' in Layer ' +str(layer_id))

    # plt.xlabel('Layer Head = '+ str(layer_id))
    # plt.ylabel('impt')

    # plt.savefig(str(layer_id)+domain+'.png')
    plt.savefig(domain+'.png')

    plt.cla()

# impt = impt.numpy()
#
# sns.distplot(impt[0], hist=True, kde=True,
#              bins=int(180/5), color = 'darkblue',
#              hist_kws={'edgecolor':'black'},
#              kde_kws={'linewidth': 4})

#
# print('impt: ',impt)

# sns.histplot(head_impt, hist=False, kde=True,
#              kde_kws={'linewidth': 3},
#              label=domain)