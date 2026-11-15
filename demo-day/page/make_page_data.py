# Sayfanın verisi: gerçek loss eğrisi, gerçek Mimi kodları, ses dosyaları. Sabah Drive'dan inenler için yeniden çalıştır.
#   python3 make_page_data.py
import json, os, shutil, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DD = os.path.dirname(HERE); DL = os.path.expanduser('~/Downloads')
exec(open(os.path.join(DD, '_build', 'src', 'common.py'), encoding='utf-8').read())
A = os.path.join(HERE, 'audio')

# 1) ön eğitim loss eğrisi (pretrain_log.json: Drive'dan indirildi)
log = json.load(open(os.path.join(DL, 'pretrain_log.json')))['log']
curve = [dict(s=l['step'], v=round(l['val'], 4), t=None if l['train'] != l['train'] else round(l['train'], 4)) for l in log]

# 2) gerçek Mimi kodları: "Merhaba, ben Alperen..." okumamın token dizisi
d = np.load(os.path.join(DD, 'tokens_me_read', 'train_read.npz')); tok, lens = d['tokens'], d['lens']
st = np.concatenate([[0], np.cumsum(lens)[:-1]]); want = norm_text('Merhaba, ben Alperen. Bu sesi ben kaydediyorum.')
seq = None
for a, n in zip(st, lens):
    q = tok[a:a + n]; k = int(np.argmax(q == SEP))
    if ''.join(CHARS[c - CHAR_BASE] for c in q[2:k]) == want: seq = q; break
k = int(np.argmax(seq == SEP)); audio = (seq[k + 1:-1].astype(int) - AUDIO_BASE).reshape(-1, N_CB)
codes = dict(text=want, frames=(audio % CB_SIZE)[:48].tolist(), n_frames=len(audio), n_tokens=int(len(seq)))

# 3) ses dosyaları: sabit adlar. Yerelde olanlar şimdi, Drive'dan inenler İndirilenler'den.
SRC = {
 'ben_gercek.wav':      [os.path.join(DD, 'my_read', 'r0001.wav')],
 'freya_0000.wav':      [os.path.join(DD, 'results', 'eval', 'pretrain_step060000', '0000.wav')],
 'codec_orijinal.wav':  [os.path.join(DD, 'codec_test', '0_original_24k.wav')],
 'codec_8.wav':         [os.path.join(DD, 'codec_test', '8_codebooks.wav')],
 'adim_000000.wav':     [os.path.join(DL, 'pretrain_step000000_0.wav')],
 'adim_001000.wav':     [os.path.join(DL, 'pretrain_step001000_0.wav')],
 'adim_003000.wav':     [os.path.join(DL, 'pretrain_step003000_0.wav')],
 'adim_010000.wav':     [os.path.join(DL, 'pretrain_step010000_0.wav')],
 'adim_060000.wav':     [os.path.join(DL, 'pretrain_step060000_0.wav')],
 'ses_200.wav':         [os.path.join(DL, 'finetune_read_step000200_0.wav')],
 'ses_1000.wav':        [os.path.join(DL, 'finetune_read_step001000_0.wav')],
}
# 04c taraması: her 100 adımda checkpoint -> WER + bana benzerlik (05b), demo cümlesi klibi, Whisper'ın duyduğu
SW = os.path.join(DD, 'results', 'sweep')
sj, heard = json.load(open(os.path.join(SW, 'sweep.json'))), json.load(open(os.path.join(SW, 'heard.json')))
sweep = dict(real=round(sj['_real']['sim'], 4), pretrain=sj['pretrain'], steps=[])
for k, v in sj.items():
    if not k.startswith('finetune_c_step'): continue
    f = f"sweep_{v['step']:04d}.wav"; SRC[f] = [os.path.join(SW, k + '.wav')]
    sweep['steps'].append(dict(s=v['step'], wer=v['wer'], sim=v['sim'], f=f, heard=heard[k]))
sweep['steps'].sort(key=lambda r: r['s'])
have = {}
for name, cands in SRC.items():
    for c in cands:
        if os.path.exists(c): shutil.copy(c, os.path.join(A, name)); break
    have[name] = os.path.exists(os.path.join(A, name))
json.dump(dict(curve=curve, codes=codes, sweep=sweep, have=have), open(os.path.join(HERE, 'data.json'), 'w'), ensure_ascii=False)
print('loss noktası', len(curve), '| kod karesi', codes['n_frames'], '| sesler:', {k: v for k, v in have.items()})
print('EKSİK:', [k for k, v in have.items() if not v] or 'yok')
