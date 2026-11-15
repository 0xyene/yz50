# %%
# YZ50 Demo Day - 5b: her checkpoint için netlik (WER) + ses benzerliği
# 04c'nin her 100 adımlık checkpoint'i: Freya-TR-Eval'in her 5. cümlesi (99 cümle), Freya protokolüyle WER/CER.
# Ses benzerliği: WavLM konuşmacı modeli (microsoft/wavlm-base-plus-sv), çıktıyı benim gerçek okumalarımla karşılaştırır (kosinüs).
# Referans da Mimi'den geçer (model çıktısı gibi), yani codec farkı ölçüme girmez. Yarıda kalırsa yeniden çalıştır: biten checkpoint'ler atlanır.

# %% [colab]
!pip -q install faster-whisper jiwer nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
# faster-whisper (CTranslate2) CUDA 12 kütüphanelerini arar; Colab'ın yeni imajında yoklar -> pip'ten gelenleri önceden yükle
import ctypes, glob, site
_libs = [p for d in site.getsitepackages() for pat in ('nvidia/cublas/lib/libcublasLt.so.12', 'nvidia/cublas/lib/libcublas.so.12', 'nvidia/cudnn/lib/libcudnn*.so.9')
         for p in sorted(glob.glob(f'{d}/{pat}'))]
for _ in range(2):                                     # cuDNN parçaları birbirine bağlı: iki tur dene
    for p in _libs:
        try: ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        except OSError: pass
print(len(_libs), 'CUDA 12 kütüphanesi yüklendi')
from google.colab import drive
drive.mount('/content/drive')

# %%
import os, re, json, time, wave, glob
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from transformers import MimiModel, AutoFeatureExtractor, WavLMForXVector
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly
from jiwer import wer, cer

# %% [include common.py]

# %% [config]
ROOT = '/content/drive/MyDrive/yz50-demo-day'
FIXED = [('pretrain', f'{ROOT}/ckpt/pretrain_last.pt', MALE), ('read_best', f'{ROOT}/ckpt/finetune_read_best.pt', ME)]
SWEEP = f'{ROOT}/ckpt/finetune_c_step*.pt'
READ_DIR = f'{ROOT}/tokens/me_read'
EVERY, N_REF = 5, 40                      # her 5. Freya cümlesi; referans için 40 okuma klibim
DEMO_TEXT = 'Merhaba, ben Alperen ve bu model benim sesimle konuşuyor.'
LIMIT = None
MAX_SECONDS, TEMP, TOP_K, SEED = 15, 0.8, 80, 0
SAVE_WAVS = 0
ASR = 'faster-whisper'
WHISPER = 'large-v3'
SV_MODEL = 'microsoft/wavlm-base-plus-sv'
DEVICE, AMP = 'cuda', True
BLOCK, N_LAYER, N_HEAD, N_EMBD, DROPOUT = 2048, 12, 12, 768, 0.0

# %% [include model.py]

# %% [include evalcore.py]

# %%
# Konuşmacı modeli + referans: okuma kayıtlarımın Mimi'den geçmiş hâli
items = items[::EVERY]
fe = AutoFeatureExtractor.from_pretrained(SV_MODEL); sv = WavLMForXVector.from_pretrained(SV_MODEL).to(DEVICE).eval()
@torch.no_grad()
def embed(wavs):
    x = [resample_poly(a, 2, 3).astype(np.float32) for a in wavs]                 # 24 kHz -> 16 kHz
    out = []
    for i in range(0, len(x), 16):
        inp = fe(x[i:i + 16], sampling_rate=16000, return_tensors='pt', padding=True).to(DEVICE)
        out.append(F.normalize(sv(**inp).embeddings, dim=-1).float().cpu())
    return torch.cat(out)
@torch.no_grad()
def decode_clips(path, n):
    d = np.load(path); t, l = d['tokens'], d['lens']; st = np.concatenate([[0], np.cumsum(l)[:-1]]); wavs = []
    for k in np.linspace(0, len(l) - 1, min(n, len(l))).astype(int):
        q = t[st[k]:st[k] + l[k]].astype(np.int64); s = int(np.argmax(q == SEP)); a = q[s + 1:-1] - AUDIO_BASE
        m = len(a) // N_CB; codes = torch.tensor(a[:m * N_CB] % CB_SIZE, device=DEVICE).view(m, N_CB).T[None]
        wavs.append(mimi.decode(codes).audio_values[0, 0].float().cpu().numpy())
    return wavs
ref = F.normalize(embed(decode_clips(f'{READ_DIR}/train_read.npz', N_REF)).mean(0), dim=-1)
sim_real = float((embed(decode_clips(f'{READ_DIR}/val.npz', 13)) @ ref).mean())
print(f'tavan: gerçek okumalarım (val, eğitimde yok) -> benzerlik {sim_real:.3f}')

# %%
# Ölç: her checkpoint
OUTJ = f'{ROOT}/eval/sweep.json'; os.makedirs(f'{ROOT}/eval/sweep', exist_ok=True)
res = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
res['_real'] = dict(sim=sim_real)
runs = FIXED + [(os.path.basename(p)[:-3], p, ME) for p in sorted(glob.glob(SWEEP))]
groups = {}
for i, it in enumerate(items): groups.setdefault(len(encode_text(it['text'])), []).append(i)
for name, path, tag in runs:
    if name in res or not os.path.exists(path): continue
    ck = torch.load(path, map_location=DEVICE)
    model = GPT(ck['cfg']).to(DEVICE).eval(); model.load_state_dict(ck['model'])
    t0, wavs = time.time(), [None] * len(items)
    for L, ids in groups.items():
        for i, a in zip(ids, speak_batch(model, [items[i]['text'] for i in ids], tag)[0]): wavs[i] = a
    refs = [normalize_tr(it['text']) for it in items]
    hyps = [normalize_tr(transcribe(a)) or '-' for a in wavs]
    sim = float((embed(wavs) @ ref).mean())
    demo = speak_batch(model, [DEMO_TEXT], tag)[0][0]; save_wav(f'{ROOT}/eval/sweep/{name}.wav', demo)
    res[name] = dict(step=ck.get('step'), tag=tag, wer=round(wer(refs, hyps), 4), cer=round(cer(refs, hyps), 4), sim=round(sim, 4), n=len(items))
    json.dump(res, open(OUTJ, 'w'), ensure_ascii=False, indent=1)
    print(f'{name:22s} WER {res[name]["wer"]:6.1%} | CER {res[name]["cer"]:6.1%} | benzerlik {sim:.3f} | {time.time() - t0:.0f} sn')
    del model; torch.cuda.empty_cache() if DEVICE == 'cuda' else None

# %%
# Tablo + grafik
rows = [(k, v) for k, v in res.items() if not k.startswith('_')]
print(f'{"checkpoint":22s} {"WER":>7s} {"CER":>7s} {"benzerlik":>10s}   (gerçek okumalarım: {res["_real"]["sim"]:.3f})')
for k, v in rows: print(f'{k:22s} {v["wer"]:7.1%} {v["cer"]:7.1%} {v["sim"]:10.3f}')
try:
    import matplotlib.pyplot as plt
    sw = sorted([v for k, v in rows if k.startswith('finetune_c_step')], key=lambda v: v['step'])
    if sw:
        fig, ax = plt.subplots(figsize=(9, 4)); ax2 = ax.twinx()
        ax.plot([v['step'] for v in sw], [100 * v['wer'] for v in sw], 'o-', color='#2b59e8'); ax.set_ylabel('WER % (düşük = net)', color='#2b59e8')
        ax2.plot([v['step'] for v in sw], [v['sim'] for v in sw], 's-', color='#e2552d'); ax2.set_ylabel('benzerlik (yüksek = bana benziyor)', color='#e2552d')
        ax2.axhline(res['_real']['sim'], color='#e2552d', ls=':', lw=1); ax.set_xlabel('fine-tune adımı')
        fig.tight_layout(); fig.savefig(f'{ROOT}/eval/sweep.png', dpi=150); plt.show()
except Exception as e: print('grafik çizilemedi:', e)
