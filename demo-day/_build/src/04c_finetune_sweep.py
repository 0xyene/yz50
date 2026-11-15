# %%
# YZ50 Demo Day - 4c: 4b'nin aynısı, daha düşük öğrenme oranı ve her 100 adımda ayrı checkpoint
# Amaç: "net" ile "bana benziyor" arasındaki adımları kaçırmamak. 05b her checkpoint'i ölçer.
# Ön eğitilmiş modelden başlar. Veri: 3 Alania dosyası + okuma kayıtlarım (15 dk, birebir metin, x20) + eski video sesim (x2). Etiket: ME.
# Drive'da gerekli: yz50-demo-day/tokens/me_read/train_read.npz + val.npz
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
RUN = 'finetune_c'
INIT_FROM = f'{ROOT}/ckpt/pretrain_last.pt'
MAX_FILES, EXTRA = 3, [(f'{ROOT}/tokens/me_read', 20), (f'{ROOT}/tokens/me', 2)]   # okuma x20, eski video sesim x2
VAL_PATH = f'{ROOT}/tokens/me_read/val.npz'           # val loss temiz okuma kayıtlarımda ölçülür
BLOCK = 2048
N_LAYER, N_HEAD, N_EMBD, DROPOUT = 12, 12, 768, 0.1
BATCH = 16
MAX_STEPS, WARMUP = 1500, 100
MAX_LR, MIN_LR, WEIGHT_DECAY = 5e-5, 5e-6, 0.1
EVAL_EVERY, VAL_BATCHES, CKPT_EVERY = 100, 20, 500
SAVE_EVERY = 100                          # her 100 adımda ckpt/finetune_c_stepNNNNN.pt (bf16, ~200 MB)
SAMPLE_TEXTS = ['Merhaba, ben Alperen ve bu model benim sesimle konuşuyor.',
                'Bu sesi kaydetmedim, bu sesi modelim üretti.',
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

