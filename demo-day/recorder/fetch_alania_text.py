# Alania (CC BY 4.0) metinlerinden aday cümle çeker: eğitimde kullanılmayan satırlar (offset >= 400k), yalnız text_spoken.
import json, random, time, subprocess
random.seed(7)
API = 'https://datasets-server.huggingface.co/rows?dataset=cloud0day3/alania-synthetic-speech-tr&config=cc-by&split=train&offset={}&length=100'
out = []
for off in random.sample(range(400_000, 1_650_000, 100), 40):
    for _ in range(3):
        try:
            d = json.loads(subprocess.run(['curl', '-sf', '--max-time', '30', API.format(off)], capture_output=True, check=True).stdout); break
        except Exception as e:
            print('retry', off, type(e).__name__); time.sleep(2)
    else: continue
    for r in d['rows']:
        x = r['row']; out.append(dict(id=x['id'], text=x['text_spoken'], cat=x['text_category'], src=x['text_source']))
json.dump(out, open('alania_candidates.json', 'w'), ensure_ascii=False)
print(len(out), 'aday')
