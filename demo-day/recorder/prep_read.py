# 03b (yerelde): okuma kayıtları -> Mimi token'ları, etiket ME. Metin = sentences.json (Whisper yok).
#   python3 prep_read.py   ->  demo-day/tokens_me_read/train_read.npz + val.npz  (Drive: yz50-demo-day/tokens/me_read/)
import os, json, math, wave
import numpy as np, torch
from scipy.signal import resample_poly
from transformers import MimiModel
HERE = os.path.dirname(os.path.abspath(__file__)); DD = os.path.dirname(HERE)
exec(open(os.path.join(DD, '_build', 'src', 'common.py'), encoding='utf-8').read())
IN, OUT = os.path.join(DD, 'my_read'), os.path.join(DD, 'tokens_me_read')
CLICK, PAD, TARGET_DBFS = 0.15, 0.10, -20.0
os.makedirs(OUT, exist_ok=True)

def runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))

def trim(x, sr):
    # baştaki tuş sesini at, sonra konuşmanın ilk ve son anına kadar kırp (03'teki enerji kuralı)
    x = x[int(CLICK * sr):]
    hop = sr // 100; n = len(x) // hop
    db = 10 * np.log10((x[:n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1) + 1e-10)
    sp = db > np.percentile(db, 10) + 12
    for s, e in runs(~sp):
        if 0 < s and e < n and e - s < 25: sp[s:e] = True       # < 0.25 sn boşluk
    for s, e in runs(sp):
        if e - s < 15: sp[s:e] = False                           # < 0.15 sn patlama (tuş sesi)
    r = runs(sp)
    if not r: return None
    a, b = max(0, r[0][0] * hop - int(PAD * sr)), min(len(x), r[-1][1] * hop + int(PAD * sr))
    return x[a:b]

def level(a):
    rms = np.sqrt((a.astype(np.float64) ** 2).mean()) + 1e-9
    return (a * min(10 ** (TARGET_DBFS / 20) / rms, 0.99 / (np.abs(a).max() + 1e-9))).astype(np.float32)

meta = [json.loads(l) for l in open(os.path.join(IN, 'metadata.jsonl'), encoding='utf-8')]
src = {s['id']: s for s in json.load(open(os.path.join(HERE, 'sentences.json'), encoding='utf-8'))}
clips = []
for m in meta:
    with wave.open(os.path.join(IN, m['id'] + '.wav')) as w:
        sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32767
    t = trim(x, sr)
    if t is None: print('ses yok, atlandı:', m['id']); continue
    a = level(resample_poly(t, SR, sr).astype(np.float32))
    clips.append(dict(id=m['id'], text=src[m['id']]['text'], own=src[m['id']]['src'] == 'own', audio=a))

mimi = MimiModel.from_pretrained('kyutai/mimi').eval()
SPF = int(SR / FRAME_RATE)
offs = AUDIO_BASE + np.arange(N_CB)[:, None] * CB_SIZE
with torch.no_grad():
    for c in clips:
        codes = mimi.encode(torch.from_numpy(c['audio'])[None, None], num_quantizers=N_CB).audio_codes[0].numpy()
        c['codes'] = (codes[:, :math.ceil(len(c['audio']) / SPF)] + offs).T.reshape(-1).astype(np.uint16)

# validation: kendi cümlelerim hariç her 20. cümle (Demo Day kelimeleri eğitimde kalsın)
alania = [c for c in clips if not c['own']]
val_ids = {c['id'] for c in alania[10::20]}
def save(sel, name):
    seqs = [np.concatenate([np.array([BOS, ME] + encode_text(c['text']) + [SEP], np.uint16), c['codes'], np.array([EOS], np.uint16)]) for c in sel]
    lens = np.array([len(s) for s in seqs], np.int32)
    np.savez(os.path.join(OUT, name), tokens=np.concatenate(seqs), lens=lens)
    return len(seqs), int(lens.sum()), sum(len(c['audio']) for c in sel) / SR / 60
tr = save([c for c in clips if c['id'] not in val_ids], 'train_read.npz')
va = save([c for c in clips if c['id'] in val_ids], 'val.npz')
print(f'train_read.npz: {tr[0]} klip, {tr[1]:,} token, {tr[2]:.1f} dk | val.npz: {va[0]} klip, {va[1]:,} token, {va[2]:.1f} dk')
raw = sum(m['seconds'] for m in meta) / 60
print(f'kırpma: {raw:.1f} dk kayıt -> {tr[2] + va[2]:.1f} dk ses')
