# %%
# YZ50 Demo Day - 1: ön eğitim verisi
# Alania Turkish Synthetic Speech, PatientDesk AI (CC BY 4.0)
# https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr
# Her klip: 48 kHz ses + metin  ->  24 kHz  ->  Mimi codec token'ları  ->  Drive'a token dosyası

# %% [colab]
from google.colab import drive
drive.mount('/content/drive')

# %%
import os, io, math, time, glob
import multiprocessing as mp
import numpy as np, torch
import soundfile as sf, pyarrow.parquet as pq
from scipy.signal import resample_poly
from huggingface_hub import hf_hub_download
from transformers import MimiModel

# %% [include common.py]

# %% [config]
ROOT = '/content/drive/MyDrive/yz50-demo-day'
REPO, SUBDIR = 'cloud0day3/alania-synthetic-speech-tr', 'data/cc-by'
VAL_SHARD, VAL_CLIPS = 0, 600            # 0. dosyanın ilk 600 klibi = validation
TRAIN_SHARDS = list(range(1, 41))        # 40 dosya ~ 300 saat. Daha fazla veri: aralığı büyüt (en çok 342)
LIMIT_CLIPS = None
MAX_SECONDS, MAX_CHARS = 16.0, 300
ENC_BATCH, CHUNK = 64, 1024
WORKERS = os.cpu_count()
DEVICE = 'cuda'
TMP = '/content/tmp_shards'
KEEP_PARQUET = False

# %%
TOK_DIR = f'{ROOT}/tokens/alania'
os.makedirs(TOK_DIR, exist_ok=True)
SPF = int(SR / FRAME_RATE)               # kare başına 1920 ses örneği

def decode_one(b):
    # flac byte'ları -> mono, 24 kHz, int16
    x, sr = sf.read(io.BytesIO(b), dtype='float32', always_2d=True)
    x = x.mean(axis=1)
    if sr != SR: x = resample_poly(x, SR, sr)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)

pool = mp.Pool(WORKERS) if WORKERS else None        # ses çözme işçileri (GPU'ya dokunmadan önce açılır)
mimi = MimiModel.from_pretrained('kyutai/mimi').to(DEVICE).eval()

@torch.no_grad()
def encode_batch(wavs):
    # wavs: int16 dizileri. Çıktı: her klip için düzleştirilmiş token dizisi (kare kare, 8 codebook)
    L = max(len(w) for w in wavs)
    x = torch.zeros(len(wavs), 1, L)
    for i, w in enumerate(wavs): x[i, 0, :len(w)] = torch.from_numpy(w.astype(np.float32) / 32767)
    codes = mimi.encode(x.to(DEVICE), num_quantizers=N_CB).audio_codes.cpu().numpy()    # (B, 8, T)
    offs = AUDIO_BASE + np.arange(N_CB)[:, None] * CB_SIZE
    out = []
    for i, w in enumerate(wavs):
        n = math.ceil(len(w) / SPF)
        out.append((codes[i, :, :n] + offs).T.reshape(-1).astype(np.uint16))
    return out

def process_shard(idx, out_path, limit, pool):
    f = hf_hub_download(REPO, f'{SUBDIR}/train-{idx:05d}.parquet', repo_type='dataset', local_dir=TMP)
    rows = pq.read_table(f, columns=['audio', 'text_spoken', 'duration', 'gender']).to_pylist()
    rows = [r for r in rows if r['duration'] <= MAX_SECONDS and 0 < len(norm_text(r['text_spoken'] or '')) <= MAX_CHARS]
    if limit: rows = rows[:limit]
    seqs = []
    for c in range(0, len(rows), CHUNK):
        part = rows[c:c + CHUNK]
        blobs = [r['audio']['bytes'] for r in part]
        wavs = pool.map(decode_one, blobs, chunksize=16) if pool else [decode_one(b) for b in blobs]
        order = np.argsort([len(w) for w in wavs])
        audio = [None] * len(wavs)
        for i in range(0, len(order), ENC_BATCH):
            ids = order[i:i + ENC_BATCH]
            for j, a in zip(ids, encode_batch([wavs[k] for k in ids])): audio[j] = a
        for r, a in zip(part, audio):
            tag = MALE if r['gender'] == 'male' else FEMALE
            head = np.array([BOS, tag] + encode_text(r['text_spoken']) + [SEP], dtype=np.uint16)
            seqs.append(np.concatenate([head, a, np.array([EOS], dtype=np.uint16)]))
    lens = np.array([len(s) for s in seqs], dtype=np.int32)
    part_path = out_path[:-4] + '.part.npz'
    np.savez(part_path, tokens=np.concatenate(seqs), lens=lens)
    os.replace(part_path, out_path)
    if not KEEP_PARQUET: os.remove(f)
    return len(rows), sum(r['duration'] for r in rows) / 3600, int(lens.sum())

# %%
# Dosya dosya işle. Yarıda kesilirse yeniden çalıştır: biten dosyalar atlanır.
jobs = [(VAL_SHARD, f'{TOK_DIR}/val.npz', VAL_CLIPS)] + [(i, f'{TOK_DIR}/train_{i:05d}.npz', LIMIT_CLIPS) for i in TRAIN_SHARDS]
t0, hours = time.time(), 0.0
for k, (idx, path, limit) in enumerate(jobs):
    if os.path.exists(path): print('var, atlandı:', os.path.basename(path)); continue
    t1 = time.time()
    n, h, ntok = process_shard(idx, path, limit, pool)
    hours += h
    print(f'{k + 1}/{len(jobs)}  {os.path.basename(path)}: {n} klip, {h:.2f} saat, {ntok:,} token | {time.time() - t1:.0f} sn (toplam {(time.time() - t0) / 60:.1f} dk, {hours:.1f} saat ses)')
if pool: pool.close()

# %%
# Özet
tot_tok = tot_clip = 0
for p in sorted(glob.glob(f'{TOK_DIR}/train_*.npz')):
    d = np.load(p); tot_tok += int(d['lens'].sum()); tot_clip += len(d['lens'])
v = np.load(f'{TOK_DIR}/val.npz')
print(f'train: {tot_clip:,} klip, {tot_tok:,} token | val: {len(v["lens"]):,} klip, {int(v["lens"].sum()):,} token')
print('en uzun örnek:', int(v['lens'].max()), 'token | sözlük:', VOCAB)
