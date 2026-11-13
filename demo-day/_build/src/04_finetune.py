# %%
# YZ50 Demo Day - 4: kendi sesime fine-tune
# Ön eğitilmiş modelden başlar. Veri: 3 Alania dosyası + kendi sesim (6 kez tekrar). Etiket: ME.
# Yarıda kesilirse baştan çalıştır: son checkpoint'ten devam eder.

# %% [colab]
from google.colab import drive
drive.mount('/content/drive')

# %%
import os, math, time, glob, json, wave
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from transformers import MimiModel

# %% [include common.py]

# %% [config]
ROOT = '/content/drive/MyDrive/yz50-demo-day'
TOK_DIR = f'{ROOT}/tokens/alania'
RUN = 'finetune_me'
INIT_FROM = f'{ROOT}/ckpt/pretrain_last.pt'
MAX_FILES, EXTRA = 3, [(f'{ROOT}/tokens/me', 6)]      # 3 alania dosyası + kendi sesim x6
VAL_PATH = f'{ROOT}/tokens/me/val.npz'                # val loss kendi sesimde ölçülür
BLOCK = 2048
N_LAYER, N_HEAD, N_EMBD, DROPOUT = 12, 12, 768, 0.1
BATCH = 16
MAX_STEPS, WARMUP = 1500, 100
MAX_LR, MIN_LR, WEIGHT_DECAY = 1e-4, 1e-5, 0.1
EVAL_EVERY, VAL_BATCHES, CKPT_EVERY = 250, 20, 500
SAMPLE_TEXTS = ['Merhaba, ben sıfırdan eğitilmiş bir konuşma modeliyim.',
                'On iki hafta önce tek bir nöronla başladım.',
                'Hesabınıza bin iki yüz elli lira yatırılmıştır.',
                'Beş bin iki yüz kırk üç numaralı dosyanız onaylandı.']
SAMPLE_TAG, SAMPLE_SECONDS, TEMP, TOP_K = ME, 8, 0.8, 80
DEVICE, COMPILE, AMP = 'cuda', True, True

# %% [include data.py]

# %% [include model.py]

# %% [include setup.py]

# %% [include speak.py]

# %% [include evalckpt.py]

# %% [include train.py]

