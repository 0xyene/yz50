# Kayıt sunucusu: sayfayı sunar, her cümlenin wav'ını my_read/ içine yazar.
#   python3 server.py            -> http://localhost:8790
import json, os, sys, wave, io
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'my_read')
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8790
os.makedirs(OUT, exist_ok=True)
SENTENCES = json.load(open(os.path.join(HERE, 'sentences.json'), encoding='utf-8'))
TEXT = {s['id']: s['text'] for s in SENTENCES}

def clip_seconds(path):
    with wave.open(path) as w: return w.getnframes() / w.getframerate()

def write_metadata():
    # my_read/metadata.jsonl: kaydı olan her cümle için id, metin, süre (03b bunu okur)
    rows = []
    for s in SENTENCES:
        p = os.path.join(OUT, f"{s['id']}.wav")
        if os.path.exists(p): rows.append(dict(id=s['id'], text=s['text'], seconds=round(clip_seconds(p), 2)))
    with open(os.path.join(OUT, 'metadata.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return rows

class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=HERE, **k)
    def log_message(self, *a): pass
    def send_json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == '/status':
            done = {r['id']: r['seconds'] for r in write_metadata()}
            return self.send_json(dict(done=done))
        if self.path.startswith('/audio/'):
            sid = self.path.split('/')[-1].split('?')[0]
            p = os.path.join(OUT, f'{sid}.wav')
            if sid in TEXT and os.path.exists(p):
                b = open(p, 'rb').read()
                self.send_response(200); self.send_header('Content-Type', 'audio/wav'); self.send_header('Content-Length', str(len(b)))
                self.end_headers(); return self.wfile.write(b)
            return self.send_json(dict(error='yok'), 404)
        return super().do_GET()
    def do_POST(self):
        if not self.path.startswith('/save/'): return self.send_json(dict(error='?'), 404)
        sid = self.path.split('/')[-1]
        if sid not in TEXT: return self.send_json(dict(error='bilinmeyen id'), 400)
        body = self.rfile.read(int(self.headers['Content-Length']))
        try:
            with wave.open(io.BytesIO(body)) as w: assert w.getnchannels() == 1 and w.getsampwidth() == 2
        except Exception as e:
            return self.send_json(dict(error=f'wav değil: {e}'), 400)
        tmp = os.path.join(OUT, f'{sid}.part.wav')
        open(tmp, 'wb').write(body); os.replace(tmp, os.path.join(OUT, f'{sid}.wav'))
        rows = write_metadata()
        return self.send_json(dict(ok=True, n=len(rows), minutes=round(sum(r['seconds'] for r in rows) / 60, 1)))

print(f'kayıt sayfası: http://localhost:{PORT}   |   kayıtlar: {OUT}')
ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()
