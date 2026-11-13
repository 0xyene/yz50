# %%
# YZ50 Demo Day - 3: kendi sesim
# 12 haftalık ödev videolarımın sesi -> sessizliklerden 1.5–12 sn parçalar -> Whisper ile metin
# -> filtre -> Mimi token'ları, etiket ME. Çıktı 04_finetune_me'nin okuduğu dosyalar.
# Drive'da gerekli: yz50-demo-day/my_voice/*.flac (24 kHz mono). A100 şart değil, T4/L4 yeter.
# Yeniden çalıştırınca Whisper sonuçları önbellekten okunur.

# %% [colab]
from google.colab import drive
drive.mount('/content/drive')

# %%
import os, re, json, glob, math, time, collections, wave
import numpy as np, torch
import soundfile as sf
from scipy.signal import resample_poly
from transformers import MimiModel, WhisperProcessor, WhisperForConditionalGeneration

# %% [include common.py]

# %% [config]
ROOT = '/content/drive/MyDrive/yz50-demo-day'
VOICE_DIR = f'{ROOT}/my_voice'
WHISPER = 'openai/whisper-large-v3'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LIMIT_SECONDS = None                     # yerel deneme için: her dosyanın ilk N saniyesi
# Sessizlik bulma (enerji tabanlı)
WIN, HOP = 0.030, 0.010                  # 30 ms pencere, 10 ms adım
THRESH_DB = 12                           # eşik = gürültü tabanı (10. yüzdelik) + 12 dB
CLOSE_GAP, MIN_BURST = 0.25, 0.15        # < 0.25 sn boşluk = konuşma devam; < 0.15 sn ses = gürültü
JOIN_GAP, MIN_SEG, MAX_SEG, EDGE = 0.8, 1.5, 12.0, 0.12   # parçaları birleştir, sınırlar, kenar payı
TARGET_DBFS = -20.0                      # her parçanın ses seviyesi eşitlenir
CPS_MIN, CPS_MAX = 4, 25                 # saniyede karakter sınırı (Whisper hatalarını eler)
VAL_FRAC = 0.03                          # her dosyanın son %3'ü validation
WHISPER_BATCH, ENC_BATCH = 16, 32
BLOCK = 2048                             # modelin bağlam uzunluğu (02/04 ile aynı)

# %%
OUT_DIR = f'{ROOT}/tokens/me'
SEG_LOG, CACHE = f'{ROOT}/my_voice_segments.jsonl', f'{ROOT}/my_voice_whisper_cache.json'
CHECK_DIR = f'{ROOT}/samples/me_check'
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(CHECK_DIR, exist_ok=True)

def load_voice(path):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(axis=1)
    if sr != SR: x = resample_poly(x, SR, sr).astype(np.float32)
    return x[:int(LIMIT_SECONDS * SR)] if LIMIT_SECONDS else x

def runs(mask):
    # True bloklarının (başlangıç, bitiş) kare indeksleri
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))

def find_segments(x):
    win, hop = int(WIN * SR), int(HOP * SR)
    n = 1 + max(0, (len(x) - win) // hop)
    fr = np.lib.stride_tricks.as_strided(x, (n, win), (x.strides[0] * hop, x.strides[0]))
    db = 10 * np.log10((fr.astype(np.float64) ** 2).mean(axis=1) + 1e-10)
    speech = db > np.percentile(db, 10) + THRESH_DB
    for s, e in runs(~speech):                                   # kısa boşlukları kapat
        if 0 < s and e < n and (e - s) * HOP < CLOSE_GAP: speech[s:e] = True
    for s, e in runs(speech):                                    # kısa patlamaları at
        if (e - s) * HOP < MIN_BURST: speech[s:e] = False
    regions = [(s * HOP, e * HOP + WIN) for s, e in runs(speech)]   # saniye
    def split_long(s, e):                                        # 12 sn'den uzunsa en sessiz yerden böl
        out = []
        while e - s > MAX_SEG:
            a, b = int((s + MIN_SEG) / HOP), int((s + MAX_SEG) / HOP)
            cut = (a + int(np.argmin(db[a:b]))) * HOP + WIN / 2
            out.append((s, cut)); s = cut
        return out + [(s, e)]
    segs, cur = [], None
    for s, e in regions:
        if cur and s - cur[1] <= JOIN_GAP and e - cur[0] <= MAX_SEG: cur = (cur[0], e)
        else:
            if cur: segs += split_long(*cur)
            cur = (s, e)
    if cur: segs += split_long(*cur)
    T = len(x) / SR
    return [(max(0.0, s - EDGE), min(T, e + EDGE)) for s, e in segs if e - s >= MIN_SEG]

def level(a):
    # parçayı TARGET_DBFS'e getir, tepe 0.99'u geçmesin
    rms = np.sqrt((a.astype(np.float64) ** 2).mean()) + 1e-9
    g = min(10 ** (TARGET_DBFS / 20) / rms, 0.99 / (np.abs(a).max() + 1e-9))
    return (a * g).astype(np.float32)

# %%
# 1) Parçalara ayır
files = sorted(glob.glob(f'{VOICE_DIR}/*.flac'), key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', os.path.basename(p))])
assert files, f'{VOICE_DIR} içinde .flac yok'
segs, total_s = [], 0.0
for p in files:
    x = load_voice(p); total_s += len(x) / SR
    name = os.path.splitext(os.path.basename(p))[0]
    found = find_segments(x)
    for s, e in found:
        segs.append(dict(file=name, start=round(float(s), 2), end=round(float(e), 2),
                         audio=level(x[int(s * SR):int(e * SR)])))
    print(f'{name}: {len(x) / SR / 60:5.1f} dk ses -> {len(found)} parça, {sum(e - s for s, e in found) / 60:5.1f} dk')
print(f'toplam: {total_s / 60:.1f} dk -> {len(segs)} parça, {sum(s["end"] - s["start"] for s in segs) / 60:.1f} dk')

# %%
# 2) Whisper ile yazıya dök (önbellekli)
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
key = lambda s: f'{WHISPER}|{s["file"]}|{s["start"]}|{s["end"]}'
todo = [s for s in segs if key(s) not in cache]
if todo:
    dt = torch.float16 if DEVICE == 'cuda' else torch.float32
    proc = WhisperProcessor.from_pretrained(WHISPER)
    asr = WhisperForConditionalGeneration.from_pretrained(WHISPER, torch_dtype=dt).to(DEVICE).eval()
    t0 = time.time()
    order = sorted(todo, key=lambda s: s['end'] - s['start'])
    for i in range(0, len(order), WHISPER_BATCH):
        part = order[i:i + WHISPER_BATCH]
        wav16 = [resample_poly(s['audio'], 2, 3) for s in part]         # 24 kHz -> 16 kHz
        feats = proc.feature_extractor(wav16, sampling_rate=16000, return_tensors='pt').input_features
        with torch.no_grad():
            ids = asr.generate(feats.to(DEVICE, dt), language='tr', task='transcribe', max_new_tokens=200)
        for s, t in zip(part, proc.batch_decode(ids, skip_special_tokens=True)): cache[key(s)] = t.strip()
        if (i // WHISPER_BATCH) % 20 == 0: print(f'whisper {i + len(part)}/{len(order)} | {time.time() - t0:.0f} sn')
    json.dump(cache, open(CACHE, 'w'), ensure_ascii=False)
    del asr; torch.cuda.empty_cache() if DEVICE == 'cuda' else None
for s in segs: s['text_raw'] = cache[key(s)]

# %%
# 3) Metni temizle ve süz
ONES = ['', 'bir', 'iki', 'üç', 'dört', 'beş', 'altı', 'yedi', 'sekiz', 'dokuz']
TENS = ['', 'on', 'yirmi', 'otuz', 'kırk', 'elli', 'altmış', 'yetmiş', 'seksen', 'doksan']
def num_tr(n):
    # tam sayıyı Türkçe okunuşuna çevir: 1250 -> "bin iki yüz elli"
    if n == 0: return 'sıfır'
    def under_1000(k):
        h, r = divmod(k, 100)
        return ' '.join(w for w in [('' if h == 1 else ONES[h]) + (' yüz' if h else ''), TENS[r // 10], ONES[r % 10]] if w.strip()).strip()
    words = []
    for size, name in [(10 ** 9, 'milyar'), (10 ** 6, 'milyon'), (1000, 'bin')]:
        q, n = divmod(n, size)
        if q: words.append(name if (q == 1 and name == 'bin') else f'{under_1000(q)} {name}')
    if n: words.append(under_1000(n))
    return ' '.join(words)

def clean(t):
    # (metin, neden): neden None ise parça kullanılır
    if not t: return None, 'boş'
    if re.search(r'\d[.,]\d', t): return None, 'ondalık sayı'
    t = re.sub(r'%\s*(\d+)', r'yüzde \1', t)
    t = re.sub(r'\d+', lambda m: num_tr(int(m.group())) if len(m.group()) <= 12 else m.group(), t)
    if re.search(r'\d', t): return None, 'sayı kaldı'
    w = t.lower().split()
    if len(w) > 4 and collections.Counter(w).most_common(1)[0][1] > 0.4 * len(w): return None, 'tekrar (Whisper hatası)'
    n = norm_text(t)
    return (n, None) if n else (None, 'boş')

reasons = collections.Counter()
for s in segs:
    s['text'], why = clean(s['text_raw'])
    dur = s['end'] - s['start']
    if why is None and not CPS_MIN <= len(s['text']) / dur <= CPS_MAX: why = 'hız sınır dışı'
    if why is None and math.ceil(dur * FRAME_RATE) * N_CB + len(s['text']) + 4 > BLOCK + 1: why = 'çok uzun'
    s['keep'], s['reason'] = why is None, why
    reasons[why or 'kullanıldı'] += 1
kept = [s for s in segs if s['keep']]
print('sonuç:', dict(reasons))
print(f'kullanılan: {len(kept)} parça, {sum(s["end"] - s["start"] for s in kept) / 60:.1f} dk')
for s in np.random.default_rng(0).permutation(np.array(kept, dtype=object))[:10]:
    print(f'  {s["file"]} {s["start"]:7.1f}s  {s["text"]}')

# %%
# 4) Mimi token'ları + train/val ayır
mimi = MimiModel.from_pretrained('kyutai/mimi').to(DEVICE).eval()
SPF = int(SR / FRAME_RATE)

@torch.no_grad()
def encode_batch(wavs):
    L = max(len(w) for w in wavs)
    x = torch.zeros(len(wavs), 1, L)
    for i, w in enumerate(wavs): x[i, 0, :len(w)] = torch.from_numpy(w)
    codes = mimi.encode(x.to(DEVICE), num_quantizers=N_CB).audio_codes.cpu().numpy()     # (B, 8, T)
    offs = AUDIO_BASE + np.arange(N_CB)[:, None] * CB_SIZE
    return [(codes[i, :, :math.ceil(len(w) / SPF)] + offs).T.reshape(-1).astype(np.uint16) for i, w in enumerate(wavs)]

order = sorted(range(len(kept)), key=lambda i: len(kept[i]['audio']))
for i in range(0, len(order), ENC_BATCH):
    ids = order[i:i + ENC_BATCH]
    for j, a in zip(ids, encode_batch([kept[k]['audio'] for k in ids])): kept[j]['codes'] = a

by_file = collections.defaultdict(list)
for s in kept: by_file[s['file']].append(s)
for lst in by_file.values():
    n_val = max(1, round(len(lst) * VAL_FRAC))
    for k, s in enumerate(lst): s['split'] = 'val' if k >= len(lst) - n_val else 'train'

def save_split(split, path):
    seqs = [np.concatenate([np.array([BOS, ME] + encode_text(s['text']) + [SEP], np.uint16), s['codes'], np.array([EOS], np.uint16)])
            for s in kept if s['split'] == split]
    lens = np.array([len(q) for q in seqs], np.int32)
    np.savez(path + '.part.npz', tokens=np.concatenate(seqs), lens=lens); os.replace(path + '.part.npz', path)
    return len(seqs), int(lens.sum())
n_tr, t_tr = save_split('train', f'{OUT_DIR}/train_me.npz')
n_va, t_va = save_split('val', f'{OUT_DIR}/val.npz')
with open(SEG_LOG, 'w', encoding='utf-8') as f:
    for s in segs:
        f.write(json.dumps({k: s.get(k) for k in ['file', 'start', 'end', 'text_raw', 'text', 'keep', 'reason', 'split']}, ensure_ascii=False) + '\n')
print(f'train_me.npz: {n_tr} parça, {t_tr:,} token | val.npz: {n_va} parça, {t_va:,} token')
print('parça listesi:', SEG_LOG)

# %%
# 5) Kulak kontrolü: 3 parça, orijinal ve Mimi'den geri çözülmüş hali
def save_wav(path, a):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
for k, s in enumerate(kept[:: max(1, len(kept) // 3)][:3]):
    codes = torch.from_numpy((s['codes'].astype(np.int64) - AUDIO_BASE).reshape(-1, N_CB).T % CB_SIZE)[None].to(DEVICE)
    with torch.no_grad(): back = mimi.decode(codes).audio_values[0, 0].float().cpu().numpy()
    save_wav(f'{CHECK_DIR}/me_{k}_orijinal.wav', s['audio']); save_wav(f'{CHECK_DIR}/me_{k}_mimi.wav', back)
    print(f'{k}: {s["text"]}')
    try:
        from IPython.display import Audio, display
        display(Audio(s['audio'], rate=SR)); display(Audio(back, rate=SR))
    except Exception: pass
