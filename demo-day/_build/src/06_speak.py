# %%
# YZ50 Demo Day - 6: modeli konuştur, ses örneğiyle (voice prompt)
# Cümlenin önüne kendi gerçek kaydımı (metin + ses token'ları) koyar; model sesi oradan devam ettirir.
# Yeni eğitim yok. Çıktı: Drive yz50-demo-day/samples/speak/*.wav

# %% [colab]
from google.colab import drive
drive.mount('/content/drive')

# %%
import os, math, wave
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from transformers import MimiModel

# %% [include common.py]

# %% [config]
ROOT = '/content/drive/MyDrive/yz50-demo-day'
CKPTS = {'best': f'{ROOT}/ckpt/finetune_read_best.pt', 'last': f'{ROOT}/ckpt/finetune_read_last.pt'}
PROMPT_FILE = f'{ROOT}/tokens/me_read/train_read.npz'
N_PROMPTS = 3                          # kaç farklı kayıt denenecek
TEXTS = ['Merhaba, ben Alperen ve bu model benim sesimle konuşuyor.',
         'Bu sesi kaydetmedim, bu sesi modelim üretti.']
SEEDS, MAX_SECONDS, TEMP, TOP_K = [0, 1], 10, 0.8, 80
DEVICE, AMP = 'cuda', True
BLOCK, N_LAYER, N_HEAD, N_EMBD, DROPOUT = 2048, 12, 12, 768, 0.0     # yalnız model.py için; asıl ayar checkpoint'ten

# %% [include model.py]

# %%
OUT = f'{ROOT}/samples/speak'; os.makedirs(OUT, exist_ok=True)
mimi = MimiModel.from_pretrained('kyutai/mimi').to(DEVICE).eval()
def autocast(): return torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=AMP and DEVICE == 'cuda')

# Ses örnekleri: okuma kayıtlarımdan 3–6 sn'lik klipler
d = np.load(PROMPT_FILE); tok, lens = d['tokens'], d['lens']; starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
def text_of(q): return ''.join(CHARS[c - CHAR_BASE] for c in q[2:int(np.argmax(q == SEP))])
cand = [k for k in range(len(lens)) if 300 + 60 <= lens[k] <= 600 + 80]
pick = [cand[j] for j in np.linspace(0, len(cand) - 1, N_PROMPTS).astype(int)]
prompts = [tok[starts[k]:starts[k] + lens[k]].astype(np.int64) for k in pick]
for k, p in enumerate(prompts): print(f'örnek {k}: {len(p)} token | "{text_of(p)}"')

@torch.no_grad()
def speak(model, text, prefix=None, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    head = ([] if prefix is None else list(prefix)) + [BOS, ME] + encode_text(text) + [SEP]
    x = torch.tensor([head], device=DEVICE); out = []
    while len(out) < int(MAX_SECONDS * TOK_PER_S) and x.shape[1] < model.c['block']:
        with autocast(): logits, _ = model(x, last_only=True)
        logits = logits[0].float(); cb = len(out) % N_CB; lo = AUDIO_BASE + cb * CB_SIZE
        mask = torch.full_like(logits, float('-inf')); mask[lo:lo + CB_SIZE] = 0
        if cb == 0 and len(out) >= N_CB: mask[EOS] = 0
        v, ix = torch.topk((logits + mask) / TEMP, TOP_K)
        t = ix[torch.multinomial(F.softmax(v, dim=-1), 1, generator=g)]
        if t.item() == EOS: break
        out.append(t.item() - lo); x = torch.cat([x, t.view(1, 1)], 1)
    n = len(out) // N_CB
    if n < 2: return np.zeros(SR // 10, np.float32)
    codes = torch.tensor(out[:n * N_CB], device=DEVICE).view(n, N_CB).T[None]
    return mimi.decode(codes).audio_values[0, 0].float().cpu().numpy()

def save_wav(path, a):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())

# %%
# Konuştur: her checkpoint × (örneksiz + her ses örneği) × her cümle × her seed
from IPython.display import Audio, display
for name, path in CKPTS.items():
    if not os.path.exists(path): print('yok:', path); continue
    ck = torch.load(path, map_location=DEVICE)
    model = GPT(ck['cfg']).to(DEVICE).eval(); model.load_state_dict(ck['model'])
    for ti, text in enumerate(TEXTS):
        for pi, pre in [(-1, None)] + list(enumerate(prompts)):
            for s in SEEDS:
                a = speak(model, text, pre, s)
                f = f'{name}_t{ti}_{"noprompt" if pi < 0 else f"prompt{pi}"}_s{s}.wav'
                save_wav(f'{OUT}/{f}', a)
                print(f'{f}  ({len(a) / SR:.1f} sn)  {text}'); display(Audio(a, rate=SR))
    del model; torch.cuda.empty_cache() if DEVICE == 'cuda' else None
print('hepsi:', OUT)
