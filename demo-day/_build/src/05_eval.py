# %%
# YZ50 Demo Day - 5: Freya-TR-Eval puanlaması
# FreyaTTS'in kendi benchmark'ı (495 Türkçe cümle), onların protokolüyle:
# sentezle -> 8 kHz banda indir -> Whisper large-v3 (faster-whisper, tr, beam 5) -> iki tarafı normalize et -> WER/CER (jiwer).
# Protokol kaynağı: github.com/freyavoiceai/FreyaTTS eval/benchmark.py. Veri: huggingface.co/datasets/freyavoice/freya-tr-eval (CC BY 4.0)
# Hangi checkpoint varsa onu puanlar; olmayanı atlar.

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
import os, re, json, time, wave
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from transformers import MimiModel
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly
from jiwer import wer, cer

# %% [include common.py]

# %% [config]
ROOT = '/content/drive/MyDrive/yz50-demo-day'
RUNS = [('finetune_read_best', f'{ROOT}/ckpt/finetune_read_best.pt', ME),     # (ad, checkpoint, ses etiketi)
        ('finetune_read_last', f'{ROOT}/ckpt/finetune_read_last.pt', ME)]
# Daha önce puanlananlar (pretrain, finetune_me_*) eval/ içindeki json'lardan tabloya eklenir; yeniden koşmaz.
LIMIT = None                              # hızlı deneme için ilk N cümle
MAX_SECONDS, TEMP, TOP_K, SEED = 15, 0.8, 80, 0
SAVE_WAVS = 20                            # ilk 20 cümlenin sesi Drive'a
ASR = 'faster-whisper'                    # olmazsa 'transformers' (aynı large-v3, beam 5)
WHISPER = 'large-v3'
DEVICE, AMP = 'cuda', True
BLOCK, N_LAYER, N_HEAD, N_EMBD, DROPOUT = 2048, 12, 12, 768, 0.0     # yalnız model.py için; asıl ayar checkpoint'ten gelir

# %% [include model.py]

# %%
# Freya'nın normalizasyonu (benchmark.py'den aynen): küçük harf, rakamları yazıya çevir, noktalamayı at
_UNITS = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
_SCALES = [(10 ** 9, "milyar"), (10 ** 6, "milyon"), (10 ** 3, "bin")]
def digits_to_words(n):
    if n == 0: return "sıfır"
    parts = []
    for scale, name in _SCALES:
        if n >= scale:
            head = n // scale; n = n % scale
            if not (scale == 1000 and head == 1): parts.append(digits_to_words(head))
            parts.append(name)
    if n >= 100:
        if n // 100 > 1: parts.append(_UNITS[n // 100])
        parts.append("yüz"); n = n % 100
    if n >= 10: parts.append(_TENS[n // 10]); n = n % 10
    if n > 0: parts.append(_UNITS[n])
    return " ".join(parts)
def normalize_tr(text):
    t = text.replace("İ", "i").replace("I", "ı").lower()
    t = re.sub(r"\d+", lambda m: digits_to_words(int(m.group(0))), t)
    t = re.sub(r"[^\wçğıöşü ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# Cümleler
path = hf_hub_download('freyavoice/freya-tr-eval', 'freya_tr_eval.jsonl', repo_type='dataset')
items = [json.loads(l) for l in open(path, encoding='utf-8')][:LIMIT]
print(len(items), 'cümle | örnek:', items[0]['text'])

# Hakem: Whisper large-v3
def to_judge_band(a):
    # 24 kHz -> 8 kHz -> 16 kHz (Freya: her sistem aynı bantta yargılanır)
    try:
        import librosa
        a8 = librosa.resample(a.astype(np.float32), orig_sr=SR, target_sr=8000)
        return librosa.resample(a8, orig_sr=8000, target_sr=16000)
    except ImportError:
        return resample_poly(resample_poly(a, 1, 3), 2, 1).astype(np.float32)
if ASR == 'faster-whisper':
    try:
        from faster_whisper import WhisperModel
        judge = WhisperModel(WHISPER, device=DEVICE, compute_type='float16' if DEVICE == 'cuda' else 'int8')
        def transcribe(a):
            segs, _ = judge.transcribe(to_judge_band(a), language='tr', beam_size=5)
            return ' '.join(s.text for s in segs).strip()
        transcribe(np.random.default_rng(0).normal(0, 0.05, SR).astype(np.float32))   # deneme: GPU kütüphaneleri gerçekten çalışıyor mu
        print('hakem: faster-whisper', WHISPER)
    except Exception as e:
        print('faster-whisper açılamadı, transformers ile devam:', repr(e)[:200]); ASR = 'transformers'
if ASR == 'transformers':
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    hf_id = WHISPER if '/' in WHISPER else f'openai/whisper-{WHISPER}'
    jdt = torch.float16 if DEVICE == 'cuda' else torch.float32
    jproc = WhisperProcessor.from_pretrained(hf_id)
    jmodel = WhisperForConditionalGeneration.from_pretrained(hf_id, torch_dtype=jdt).to(DEVICE).eval()
    def transcribe(a):
        f = jproc.feature_extractor(to_judge_band(a), sampling_rate=16000, return_tensors='pt').input_features
        with torch.no_grad(): ids = jmodel.generate(f.to(DEVICE, jdt), language='tr', task='transcribe', num_beams=5, max_new_tokens=200)
        return jproc.batch_decode(ids, skip_special_tokens=True)[0].strip()

mimi = MimiModel.from_pretrained('kyutai/mimi').to(DEVICE).eval()
def autocast(on=True):
    return torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=AMP and on and DEVICE == 'cuda')

# %%
# Hızlı konuşturma: KV önbellekli, aynı metin uzunluğundaki cümleler tek batch'te.
# Örnekleme kuralı 02/04'teki speak() ile aynı (codebook maskesi, EOS yalnız kare sınırında, top-k).
@torch.no_grad()
def lm_step(model, idx, cache, pos):
    # idx: (B, T) yeni token'lar, pos: ilk yeni token'ın konumu. cache: katman başına önceden ayrılmış (K, V)
    B, T = idx.shape
    C, nh = model.c['n_embd'], model.c['n_head']
    x = model.wte(idx) + model.wpe(torch.arange(pos, pos + T, device=idx.device))
    for li, b in enumerate(model.h):
        q, k, v = b.attn.c_attn(b.ln_1(x)).split(C, dim=2)
        q, k, v = [t.view(B, T, nh, C // nh).transpose(1, 2) for t in (q, k, v)]
        K, V = cache[li]
        K[:, :, pos:pos + T], V[:, :, pos:pos + T] = k, v
        y = F.scaled_dot_product_attention(q, K[:, :, :pos + T], V[:, :, :pos + T], is_causal=T > 1)
        x = x + b.attn.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))
        x = x + b.mlp(b.ln_2(x))
    return model.lm_head(model.ln_f(x[:, -1])).float()

def new_cache(model, B, dtype):
    c = model.c; hs = c['n_embd'] // c['n_head']
    return [tuple(torch.zeros(B, c['n_head'], c['block'], hs, device=DEVICE, dtype=dtype) for _ in range(2)) for _ in range(c['n_layer'])]

@torch.no_grad()
def speak_batch(model, texts, tag, seconds=MAX_SECONDS, temp=TEMP, top_k=TOP_K, seed=SEED, amp=True):
    prompts = [[BOS, tag] + encode_text(t) + [SEP] for t in texts]
    assert len({len(p) for p in prompts}) == 1, 'aynı uzunlukta metinler'
    B, P = len(prompts), len(prompts[0])
    n_max = min(int(seconds * TOK_PER_S), model.c['block'] - P) // N_CB * N_CB
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    with autocast(amp):
        dtype = torch.bfloat16 if (AMP and amp and DEVICE == 'cuda') else torch.float32
        cache = new_cache(model, B, dtype)
        logits = lm_step(model, torch.tensor(prompts, device=DEVICE), cache, 0)
        out = torch.zeros(B, n_max, dtype=torch.long, device=DEVICE)
        done = torch.full((B,), n_max, dtype=torch.long, device=DEVICE)       # her satırın ses uzunluğu
        for n in range(n_max):
            cb = n % N_CB; lo = AUDIO_BASE + cb * CB_SIZE
            mask = torch.full_like(logits, float('-inf')); mask[:, lo:lo + CB_SIZE] = 0
            if cb == 0 and n >= N_CB: mask[:, EOS] = 0
            v, ix = torch.topk((logits + mask) / temp, top_k)
            tok = ix.gather(1, torch.multinomial(F.softmax(v, dim=-1), 1, generator=g))
            ended = (tok[:, 0] == EOS) & (done == n_max)
            done[ended] = n
            out[:, n] = tok[:, 0]
            if bool((done < n_max).all()): break
            logits = lm_step(model, tok, cache, P + n)
    wavs = []
    for i in range(B):
        n = int(done[i]) // N_CB
        if n < 2: wavs.append(np.zeros(SR // 10, np.float32)); continue
        codes = (out[i, :n * N_CB] - AUDIO_BASE).view(n, N_CB).T % CB_SIZE
        wavs.append(mimi.decode(codes[None]).audio_values[0, 0].float().cpu().numpy())
    return wavs, [int(d) < n_max for d in done]

@torch.no_grad()
def check_cache(model, text, tag, n=48):
    # Doğruluk: önbellekli üretim = önbelleksiz üretim (greedy, fp32, ilk n token)
    x = torch.tensor([[BOS, tag] + encode_text(text) + [SEP]], device=DEVICE)
    slow = []
    for k in range(n):
        logits, _ = model(x, last_only=True)
        lo = AUDIO_BASE + (k % N_CB) * CB_SIZE
        t = lo + int(logits[0, lo:lo + CB_SIZE].argmax()); slow.append(t)
        x = torch.cat([x, torch.tensor([[t]], device=DEVICE)], 1)
    P = x.shape[1] - n
    cache = new_cache(model, 1, torch.float32)
    logits = lm_step(model, x[:, :P], cache, 0); fast = []
    for k in range(n):
        lo = AUDIO_BASE + (k % N_CB) * CB_SIZE
        t = lo + int(logits[0, lo:lo + CB_SIZE].argmax()); fast.append(t)
        logits = lm_step(model, torch.tensor([[t]], device=DEVICE), cache, P + k)
    same = sum(a == b for a, b in zip(slow, fast))
    print(f'KV önbellek kontrolü: {same}/{n} token aynı')
    return same == n

def save_wav(path, a):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())

# %%
# Puanla
os.makedirs(f'{ROOT}/eval', exist_ok=True)
groups = {}
for i, it in enumerate(items): groups.setdefault(len(encode_text(it['text'])), []).append(i)
summary = {}
for name, ckpt, tag in RUNS:
    if not os.path.exists(ckpt): print('yok, atlandı:', ckpt); continue
    ck = torch.load(ckpt, map_location=DEVICE)
    model = GPT(ck['cfg']).to(DEVICE).eval(); model.load_state_dict(ck['model'])
    print(f'\n=== {name}: adım {ck.get("step")} ===')
    check_cache(model, items[0]['text'], tag)
    wav_dir = f'{ROOT}/eval/{name}_step{ck.get("step", 0):06d}'; os.makedirs(wav_dir, exist_ok=True)
    rows, t_syn, t_audio, t0 = [None] * len(items), 0.0, 0.0, time.time()
    for L, ids in sorted(groups.items()):
        t1 = time.time()
        wavs, ended = speak_batch(model, [items[i]['text'] for i in ids], tag)
        t_syn += time.time() - t1
        for i, a, e in zip(ids, wavs, ended):
            t_audio += len(a) / SR
            ref, hyp_raw = normalize_tr(items[i]['text']), transcribe(a)
            hyp = normalize_tr(hyp_raw)
            rows[i] = dict(id=items[i]['id'], text=items[i]['text'], hyp=hyp_raw, wer=wer(ref, hyp) if hyp else 1.0,
                           cer=cer(ref, hyp) if hyp else 1.0, seconds=round(len(a) / SR, 2), eos=e)
            if i < SAVE_WAVS: save_wav(f'{wav_dir}/{items[i]["id"]}.wav', a)
        k = sum(r is not None for r in rows)
        print(f'{k}/{len(items)} cümle | {time.time() - t0:.0f} sn')
    refs = [normalize_tr(r['text']) for r in rows]
    hyps = [normalize_tr(r['hyp']) or '-' for r in rows]
    res = dict(run=name, step=ck.get('step'), tag=tag, params=sum(p.numel() for p in model.parameters()), sentences=len(rows), asr=ASR,
               wer=round(wer(refs, hyps), 4), cer=round(cer(refs, hyps), 4), rtf=round(t_syn / max(t_audio, 1e-6), 3),
               eos_rate=round(np.mean([r['eos'] for r in rows]), 3), rows=rows)
    json.dump(res, open(f'{ROOT}/eval/{name}_freya_tr_eval.json', 'w'), ensure_ascii=False, indent=1)
    summary[name] = res
    print(f'{name}: WER {res["wer"]:.1%} | CER {res["cer"]:.1%} | cümlesini bitirdi (EOS) {res["eos_rate"]:.0%} | RTF {res["rtf"]}')
    for r in sorted(rows, key=lambda r: r['wer'])[:3] + sorted(rows, key=lambda r: -r['wer'])[:2]:
        print(f'  wer {r["wer"]:.2f} | {r["text"]}  ->  {r["hyp"]}')
    del model; torch.cuda.empty_cache() if DEVICE == 'cuda' else None

# %%
# Tablo (Freya'nın README'sindeki sayılarla yan yana)
table = [('FreyaTTS-small (183M)', 0.080, 0.030), ('XTTS-v2 (0.5B)', 0.111, None), ('F5-TTS (0.3B)', 0.243, None)]
import glob as _g
for p in sorted(_g.glob(f'{ROOT}/eval/*_freya_tr_eval.json')):
    r = json.load(open(p)); summary.setdefault(r['run'], r)
table += [(f'Benim modelim: {n} ({r.get("params", 99310080) / 1e6:.0f}M, adım {r["step"]})', r['wer'], r['cer']) for n, r in summary.items()]
print(f'{"model":45s} {"WER":>7s} {"CER":>7s}')
for n, w, c in table: print(f'{n:45s} {w:7.1%} {(f"{c:.1%}" if c is not None else "-"):>7s}')
json.dump([dict(model=n, wer=w, cer=c) for n, w, c in table], open(f'{ROOT}/eval/table.json', 'w'), ensure_ascii=False, indent=1)
