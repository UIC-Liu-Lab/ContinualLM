
  
# ContinualLM    
 <p align="center">    
    <br>    
    <a href="https://github.com/UIC-Liu-Lab//ContinualLM">    
        <img src="https://github.com/UIC-Liu-Lab/ContinualLM/blob/main/docs/icon.png" width="200"/>    
    </a>       
        <figcaption>Imagine an LM that not only effortlessly acquires new knowledge but also retains its mastery of skills, all while successfully transferring knowledge. Is it even possible?</figcaption>  
    <br>    
<p>
  
## Quick Links  
  
 - [Introduction ](#Introduction)  
 - [Dataset](#dataset)  
 - [Architecture](#architecture)  
 - [Installation](#installation)  
 - [Domain-adaptive Pre-training](#domain-adaptive-pre-training)  
 - [End-task Fine-tuning](#end-task-fine-tuning)  
 - [Checkpoints in Huggingface](#checkpoints-in-huggingface)  
 - [Reference](#reference)  
 - [Contact](#contact)  
  
## Introduction    
 In 2021, we introduced  [Pycontinual](https://github.com/ZixuanKe/PyContinual), a straightforward and flexible framework for continual learning. Our research has benefited significantly from this framework. Today, we are excited to announce the launch of **ContinualLM**, an extensible continual learning framework focused on language models (LMs), designed to sustain the benefits of continual learning (CL) in this field.    
    
Continual learning for LMs is distinct from traditional CL because     
 - Each task is treated as a **domain-specific corpus** (at present, our primary focus is on domain-adaptive pre-training, which is also known as pre-finetuning or post-training).  
 - Moreover, the evaluation process involves **fine-tuning** the corresponding end-task.    
    
Our repository includes a PyTorch implementation of a collection of state-of-the-art (SoTA) methods, using the same training and evaluation pipeline. This repository is committed to advancing the field of continual learning for LMs. The methods included are:    
    
    
* From our group:
   * **DAS**: [Continual Learning of Language Models](https://arxiv.org/abs/2210.05549), ICLR 2023    
   * **CPT**: [Continual Training of Language Models for Few-Shot Learning](https://arxiv.org/abs/2210.05549), EMNLP 2022    
   * **DGA**: [Adapting a Language Model While Preserving its General Knowledge](https://arxiv.org/abs/2301.08986), EMNLP 2022    
   * **CTR**: [Achieving Forgetting Prevention and Knowledge Transfer in Continual Learning](https://proceedings.neurips.cc/paper/2021/hash/bcd0049c35799cdf57d06eaf2eb3cff6-Abstract.html), NeurIPS 2021  
   * **CLASSIC**: [CLASSIC: Continual and Contrastive Learning of Aspect Sentiment Classification Tasks](https://aclanthology.org/2021.emnlp-main.550/), EMNLP 2021    
   * **B-CL**: [Adapting BERT for Continual Learning of a Sequence of Aspect Sentiment Classification Tasks](https://www.aclweb.org/anthology/2021.naacl-main.378.pdf), NAACL 2021   
   
* From other groups **(more to come)**:
  * **DEMIX**: [Demix layers: Disentangling domains for modular language modeling](https://aclanthology.org/2022.naacl-main.407);, Gururangan et al., NAACL 2022)  
  * **EWC**: [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796), Kirkpatrick et al., PNAS 2017    
  * **DER++**: [Dark experience for general continual learning: a strong, simple baseline](https://arxiv.org/abs/2004.07211), Buzzega et al., NeuriPS 2020    
  * **HAT**: [Overcoming catastrophic forgetting with hard attention to the task](http://proceedings.mlr.press/v80/serra18a.html), Serrà et al., ICML 2018    
       
* Widely employed baselines for continual learning:
  * **NCL**: Naive continual learning:continual domain-adaptive pre-training of a sequence of domains, without any specific attention paid to the issues of forgetting or transfer.
  * **ONE**: Individually conducting domain-adaptive pre-training for each domain.
  * **Adapter-ONE**: Adds adapter to Transformer for each domain  
  * **Prompt-ONE**:  Adds prompt to Transformer for each domain  
  * **KD**: Naive knoweldge Distillation  


## Dataset  
  
When it comes to the continual learning of language models (LMs), finding appropriate datasets is crucial. The datasets we provide adhere to the following principles:  
  
*  **Domain-specific:** The domain corpus must be specific enough to enhance end-task performance.  
*  **End-task available**: We favor assessing the trained language models through the end-task rather than relying on perplexity, since the former represents a more dependable evaluation approach.  

We release our dataset comprising **6** distinct domains, each accompanied by its corresponding end-task. The dataset can be found [here](https://drive.google.com/file/d/1_fAu9dPHUpFyAbAN1aBByib3tEVRlZpS/view?usp=sharing). Below are some statistics for each domain:

| Domain Corpus    | Size   | End-task    | Task                                     | #Training | #Testing | #Classes |
|------------------|--------|-------------|------------------------------------------|-----------|----------|----------|
| Yelp Restaurant  | 758MB  | Restaurant  | Aspect Sentiment Classification (ASC)    | 3,452     | 1,120    | 3        |
| Amazon Phone     | 724MB  | Phone       | Aspect Sentiment Classification (ASC)    | 239       | 553      | 2        |
| Amazon Camera    | 319MB  | Camera      | Aspect Sentiment Classification (ASC)    | 230       | 626      | 2        |
| ACL Papers       | 867MB  | ACL         | Citation Intent Classification           | 1,520     | 421      | 6        |
| AI Papers        | 507MB  | AI          | Relation Classification                  | 2,260     | 2,388    | 7        |
| PubMed Papers    | 989MB  | PubMed      | Chemical-protein Interaction Prediction  | 2,667     | 7,398    | 13       |


  
## Architecture  
The architecture of ContinualLM largely follows that of  [Pycontinual](https://github.com/ZixuanKe/PyContinual), [CPT](https://github.com/UIC-Liu-Lab/CPT) and [DGA](https://github.com/UIC-Liu-Lab/DGA).

## Installation

```conda create --name continuallm --file requirements.txt```

:warning: Our model is based on `transformers==4.17.0` and `adapter-transformers==3.0.1`. We recommend using these specific versions, as using other versions may result in unexpected bugs.
  
## Domain-adaptive Pre-training  
This is where continual learning happens. We will learn a sequnce of domains.   
  
```bash  
max_samples=640000 
for idrandom in 0 
do    
 for pt_task in 0 1 2 3 4 5    
  do    
 python -m torch.distributed.launch --nproc_per_node 4 --use_env posttrain.py \    
 --per_device_train_batch_size 62 \ 
 --fp16\    
 --max_seq_length 164 \ 
 --max_samples ${max_samples} \ 
 --idrandom ${idrandom} \ 
 --ntasks 6 \ 
 --pt_task ${pt_task} \ 
 --baseline 'das'
 done 
done  
```  
* `--idrandom`: choose the task sequence. See `./sequences` for more details.  
* `--baseline`: see the introduction for available baseline models (see ```choices``` in ```config.py```).  
  
## End-task Fine-tuning  
After conitinual learning of LMs, now we are able to evaluate the performace by runing end-task fine-tuning **individually**.  
```bash  
max_samples=640000    
 seed=(2021 111 222 333 444 555 666 777 888 999)    
 for round in 0; do    
  for idrandom in 0;    
  do    
    for pt_task in 0 1 2 3 4 5   
    do    
      for ft_task in $(seq 0 ${pt_task});    
      do    
       python finetune.py \    
       --max_seq_length 164 \ 
       --pt_task ${pt_task} \ 
       --ft_task ${ft_task} \ 
       --idrandom ${idrandom} \ 
       --ntasks 6 \ 
       --max_samples ${max_samples} \
       --seed ${seed[$round]} \ 
       --baseline 'das'    
       done    
    done   
  done  
done  
```  
  
  
## Checkpoints in Huggingface  

[comment]: <> (For those who are interested solely in the resulting model or want to continue per-training the model with their own data, we have good news! We offer checkpoints through Hugging Face:  )
  
**[TODO]**  
  
## Reference  
We highly appreciate your act of staring and citing. Your attention to detail and recognition is greatly valued.  
  
  
```bibtex  
  
@inproceedings{ke2022dgs,  
 title={Continual Learning of Language Models}, author={Ke, Zixuan and Shao, Yijia and Lin, Haowei and Konishi, Tatsuya and Kim, Gyuhak and Liu, Bing}, booktitle={International Conference on Learning Representations (ICLR)}, year={2023}}  
  
@inproceedings{ke2022dga,  
 title={Adapting a Language Model While Preserving its General Knowledge}, author={Ke, Zixuan and Shao, Yijia and Lin, Haowei and Xu, Hu and Shu, Lei, and Liu, Bing}, booktitle={Empirical Methods in Natural Language Processing (EMNLP)}, year={2022}}  
  
@inproceedings{ke2022continual,  
 title={Continual Training of Language Models for Few-Shot Learning}, author={Ke, Zixuan and Lin, Haowei and Shao, Yijia and Xu, Hu and Shu, Lei, and Liu, Bing}, booktitle={Empirical Methods in Natural Language Processing (EMNLP)}, year={2022}}  
```  
  
  
## Contact

If you have any questions regarding the code, please feel free to send an email to [Zixuan Ke](https://vincent950129.github.io/), [Yijia Shao](https://shaoyijia.github.io/), or [Haowei Lin](https://linhaowei1.github.io/). Alternatively, you may open an issue. We would like to express our gratitude to [Bing Liu](https://www.cs.uic.edu/~liub/), [Hu Xu](https://howardhsu.github.io/), and [Lei Shu](https://leishu02.github.io/) for their valuable comments and opinions 

[comment]: <> (With gratitude, we want to acknowledge that the creation of this repository would not have been possible without the invaluable contributions of [Zixuan Ke]&#40;https://vincent950129.github.io/&#41;, [Yijia Shao]&#40;https://shaoyijia.github.io/&#41;, [Haowei Lin]&#40;https://linhaowei1.github.io/&#41;, [Hu Xu]&#40;https://howardhsu.github.io/&#41;, [Lei Shu]&#40;https://leishu02.github.io/&#41;, and [Bing Liu]&#40;https://www.cs.uic.edu/~liub/&#41;.)
