# Neural Language Modeling

## Pre-trained models

Description | Parameters | Dataset | Model and Test set(s)
---|---:|---|---
Adaptive Inputs <br> ([Baevski and Auli, 2018](https://arxiv.org/abs/1809.10853)) | 1026M | [Google Billion Words](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_gbw_huge.tar.bz2)
Adaptive Inputs <br> ([Baevski and Auli, 2018](https://arxiv.org/abs/1809.10853)) | 247M | [WikiText-103](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_wiki103.tar.bz2)


## Example usage

Interactive generation via PyTorch Hub:
```
>>> import torch
>>> lm = torch.hub.load(
...   'pytorch/fairseq',
...   'transformer_lm',
...   model_name_or_path='transformer_lm.wiki103.adaptive',
...   data_name_or_path='./data-bin',
...   tokenizer='moses',
...   aggressive_dash_splits=True,
...   no_escape=True,
...   beam=1,
...   sampling=True,
...   sampling_topk=10,
...   temperature=0.8,
... )
>>> lm.generate('Barack Obama', verbose=True)
```

Available models are listed in the ``hub_models()`` method in each model file, for example:
[transformer_lm.py](https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer_lm.py).


## Training a new model with the CLI tools

These scripts provide an example of pre-processing data for the Language Modeling task.

### prepare-wikitext-103.sh

Provides an example of pre-processing for [WikiText-103 language modeling task](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/):

Example usage:

Prepare data:
```
$ cd examples/language_model/
$ bash prepare-wikitext-103.sh
$ cd ../..

# Binarize the dataset:
$ TEXT=examples/language_model/wikitext-103

$ fairseq-preprocess --only-source \
  --trainpref $TEXT/wiki.train.tokens --validpref $TEXT/wiki.valid.tokens --testpref $TEXT/wiki.test.tokens \ 
  --destdir data-bin/wikitext-103
```

Train a transformer language model with adaptive inputs ([Baevski and Auli (2018): Adaptive Input Representations for Neural Language Modeling](transformer_lm/README.md)):
```
# If it runs out of memory, try to reduce max-tokens and tokens-per-sample
$ mkdir -p checkpoints/transformer_wikitext-103
$ fairseq-train --task language_modeling data-bin/wikitext-103 \
  --save-dir checkpoints/transformer_wikitext-103 --arch transformer_lm_wiki103 \
  --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
  --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
  --criterion adaptive_loss --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d

# Evaluate:
$ fairseq-eval-lm data-bin/wikitext-103 --path 'checkpoints/transformer_wiki103/checkpoint_best.pt' \
  --sample-break-mode complete --max-tokens 3072 --context-window 2560 --softmax-batch 1024
```

Train a convolutional language model ([Dauphin et al. (2017): Language Modeling with Gated Convolutional Networks](conv_lm/README.md)):
```
# If it runs out of memory, try to reduce max-tokens and tokens-per-sample
$ mkdir -p checkpoints/fconv_wikitext-103
$ fairseq-train --task language_modeling data-bin/wikitext-103 \
  --save-dir checkpoints/fconv_wikitext-103 \
  --max-epoch 35 --arch fconv_lm_dauphin_wikitext103 --optimizer nag \
  --lr 1.0 --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 \
  --clip-norm 0.1 --dropout 0.2 --weight-decay 5e-06 --criterion adaptive_loss \
  --adaptive-softmax-cutoff 10000,20000,200000 --max-tokens 1024 --tokens-per-sample 1024
  --ddp-backend=no_c10d

# Evaluate:
$ fairseq-eval-lm data-bin/wikitext-103 --path 'checkpoints/fconv_wiki103/checkpoint_best.pt'
```
