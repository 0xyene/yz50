# Modeli bilgisayarımda konuştur: sayfadan cümle gelir, model sesi üretir, WAV döner.
#   python3 speak/server.py            ->  http://127.0.0.1:8793
# Checkpoint'ler: demo-day/speak/ckpt/*.pt (04c'nin bf16 adım dosyaları, Drive'dan indirilir)
import os, io, sys, json, glob, time, wave, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from transformers import MimiModel

HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(HERE, '..', '_build', 'src')
BLOCK, N_LAYER, N_HEAD, N_EMBD, DROPOUT = 2048, 12, 12, 768, 0.0      # yalnız model.py için; asıl ayar checkpoint'ten
exec(open(os.path.join(SRC, 'common.py'), encoding='utf-8').read())
exec(open(os.path.join(SRC, 'model.py'), encoding='utf-8').read())
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8793
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
MAX_SECONDS, TEMP, TOP_K = 12, 0.8, 80
TAGS = {'ME': ME, 'MALE': MALE, 'FEMALE': FEMALE}
import re
_EV = open(os.path.join(SRC, 'evalcore.py'), encoding='utf-8').read()
exec(_EV[_EV.index('_UNITS'):_EV.index('def normalize_tr')])          # digits_to_words (Freya'nın kuralı)
def spell_numbers(t):                                                  # model rakam görmedi: 50 -> elli
    return re.sub(r'\d+', lambda m: ' ' + digits_to_words(int(m.group(0))) + ' ', t)

mimi = MimiModel.from_pretrained('kyutai/mimi').eval()                # çözücü CPU'da
models, lock = {}, threading.Lock()

def ckpt_files():
    # speak/ckpt/*.pt ve İndirilenler'deki 04c adım dosyaları (Drive'dan inince taşımaya gerek yok)
    ps = glob.glob(os.path.join(HERE, 'ckpt', '*.pt')) + glob.glob(os.path.expanduser('~/Downloads/finetune_c_step*.pt'))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in sorted(ps, key=os.path.basename)}

def get_model(name):
    if name not in models:
        ck = torch.load(ckpt_files()[name], map_location='cpu')
        m = GPT(ck['cfg']); m.load_state_dict({k: v.float() for k, v in ck['model'].items()})
        models[name] = m.to(DEVICE).eval()
        print(f'yüklendi: {name} (adım {ck.get("step")})', flush=True)
    return models[name]

@torch.no_grad()
def lm_step(model, idx, cache, pos):
    # 05'teki KV önbellekli adım: yalnız yeni token'ları işler
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

@torch.no_grad()
def speak(model, text, tag, seed=0, temp=TEMP, top_k=TOP_K):
    # örnekleme kuralı eğitimdekiyle aynı: codebook maskesi, EOS yalnız kare sınırında, top-k
    prompt = [BOS, tag] + encode_text(text) + [SEP]
    c = model.c; hs = c['n_embd'] // c['n_head']; P = len(prompt)
    n_max = min(int(MAX_SECONDS * TOK_PER_S), c['block'] - P) // N_CB * N_CB
    cache = [tuple(torch.zeros(1, c['n_head'], c['block'], hs, device=DEVICE) for _ in range(2)) for _ in range(c['n_layer'])]
    g = torch.Generator(device='cpu').manual_seed(seed)
    logits = lm_step(model, torch.tensor([prompt], device=DEVICE), cache, 0)
    out, ended = [], False
    for n in range(n_max):
        cb = n % N_CB; lo = AUDIO_BASE + cb * CB_SIZE
        mask = torch.full_like(logits, float('-inf')); mask[:, lo:lo + CB_SIZE] = 0
        if cb == 0 and n >= N_CB: mask[:, EOS] = 0
        v, ix = torch.topk((logits + mask) / temp, top_k)
        pick = torch.multinomial(F.softmax(v, dim=-1).cpu(), 1, generator=g).to(DEVICE)
        tok = ix.gather(1, pick)
        if tok.item() == EOS: ended = True; break
        out.append(tok.item() - lo)
        logits = lm_step(model, tok, cache, P + n)
    n = len(out) // N_CB
    if n < 2: return np.zeros(SR // 10, np.float32), ended
    codes = torch.tensor(out[:n * N_CB]).view(n, N_CB).T[None]
    return mimi.decode(codes).audio_values[0, 0].float().numpy(), ended

def warm(names):
    with lock:
        for n in names:
            if n.startswith('finetune_c') and n not in models: get_model(n)

def wav_bytes(a):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()

class H(BaseHTTPRequestHandler):
    def cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Allow-Private-Network', 'true')
        self.send_header('Access-Control-Expose-Headers', 'X-Info')
    def do_OPTIONS(self):
        self.send_response(204); self.cors(); self.end_headers()
    def send_json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.cors(); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith('/status'):
            names = list(ckpt_files())
            if names and not models: threading.Thread(target=warm, args=(names,), daemon=True).start()   # ilk cümle beklemesin
            return self.send_json(200, dict(ok=True, device=DEVICE, ckpts=names))
        self.send_json(404, dict(error='yok'))
    def do_POST(self):
        if not self.path.startswith('/speak'): return self.send_json(404, dict(error='yok'))
        try:
            q = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            text = ' '.join(spell_numbers(str(q.get('text', ''))).split())[:300]
            names = ckpt_files(); name = q.get('ckpt') or next((k for k in reversed(list(names)) if k.startswith('finetune_c')), next(iter(names), None))
            if not encode_text(text): return self.send_json(400, dict(error='boş metin'))
            if name not in names: return self.send_json(400, dict(error='checkpoint yok', ckpts=list(names)))
            t0 = time.time()
            with lock:
                a, ended = speak(get_model(name), text, TAGS.get(q.get('tag', 'ME'), ME), seed=int(q.get('seed', 0)), temp=float(q.get('temp', TEMP)))
            info = dict(ckpt=name, seconds=round(len(a) / SR, 2), gen=round(time.time() - t0, 2), ended=ended, text=norm_text(text))
            b = wav_bytes(a)
            self.send_response(200); self.cors(); self.send_header('Content-Type', 'audio/wav')
            self.send_header('X-Info', json.dumps(info)); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
            print('konuştu:', info, flush=True)
        except Exception as e:
            self.send_json(500, dict(error=repr(e)))
    def log_message(self, *a): pass

if __name__ == '__main__':
    print(f'cihaz {DEVICE} | checkpoint\'ler: {list(ckpt_files()) or "YOK (speak/ckpt/ klasörüne .pt koy)"}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()
