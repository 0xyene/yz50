# %%
# YZ50 Demo Day - 2: sıfırdan konuşma modeli (ön eğitim)
# Tek bir GPT: metin karakterlerini okur, ardından ses token'larını tek tek tahmin eder.
# Veri: Alania Turkish Synthetic Speech, PatientDesk AI (CC BY 4.0). Codec: Mimi (kyutai).
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
RUN = 'pretrain'
INIT_FROM = None                          # fine-tune için: başka bir koşunun checkpoint yolu
MAX_FILES, EXTRA = None, []               # MAX_FILES: kaç alania dosyası (None = hepsi). EXTRA: [(dizin, tekrar)]
VAL_PATH = f'{TOK_DIR}/val.npz'
BLOCK = 2048
N_LAYER, N_HEAD, N_EMBD, DROPOUT = 12, 12, 768, 0.0
BATCH = 16
MAX_STEPS, WARMUP = 30000, 1000
MAX_LR, MIN_LR, WEIGHT_DECAY = 5e-4, 5e-5, 0.1
EVAL_EVERY, VAL_BATCHES, CKPT_EVERY = 1000, 20, 2000
SAMPLE_TEXTS = ['Merhaba, ben sıfırdan eğitilmiş bir konuşma modeliyim.',
                'On iki hafta önce tek bir nöronla başladım.',
                # FreyaTTS raporundaki iki sabit cümle (aynı sözlerle kulaktan karşılaştırma için)
                'Hesabınıza bin iki yüz elli lira yatırılmıştır.',
                'Beş bin iki yüz kırk üç numaralı dosyanız onaylandı.']
SAMPLE_TAG, SAMPLE_SECONDS, TEMP, TOP_K = MALE, 8, 0.8, 80
DEVICE, COMPILE, AMP = 'cuda', True, True

# %% [include data.py]

# %% [include model.py]

# %% [include setup.py]

# %% [include speak.py]

# %% [include evalckpt.py]

# %% [include train.py]

