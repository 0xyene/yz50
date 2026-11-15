# Okuma metni: Alania adaylarından süzülmüş cümleler + Demo Day kelimelerini içeren kendi cümlelerim.
# Freya-TR-Eval ile örtüşen hiçbir cümle girmez (birebir, içerme ya da ortak 5 kelimelik dizi).
import json, re, random, unicodedata, subprocess, os, collections
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, '..', '_build', 'src', 'common.py'), encoding='utf-8').read())   # CHARS, norm_text
N_TOTAL, N_MIN = 500, 300
random.seed(1337)

OWN = """Merhaba, ben Alperen. Bu sesi ben kaydediyorum.
On iki hafta boyunca her hafta bir sinir ağını sıfırdan yazdım.
YZ elli dersinde ilk hafta tek bir nöronla başladık.
Bir nöron girdileri ağırlıklarla çarpar ve sonucu toplar.
Geri yayılım, hatanın her ağırlığa ne kadar bağlı olduğunu hesaplar.
Kayıp düştükçe model veriyi daha iyi tahmin etmeye başlar.
Önce harfleri sayarak basit bir dil modeli kurdum.
Sonra çok katmanlı bir ağ ile isim üreten bir model eğittim.
Dikkat mekanizması her kelimenin diğer kelimelere bakmasını sağlar.
Transformer, dikkat katmanlarının üst üste dizilmesiyle oluşur.
Token, modelin okuduğu en küçük metin parçasıdır.
Bu model metni okuyor ve ardından sesi tahmin ediyor.
Ses, codec ile saniyede yüz küçük parçaya dönüştürülüyor.
Modeli Colab'de tek bir ekran kartıyla eğittim.
Ön eğitimde üç yüz elli saatten fazla konuşma kullandım.
Doğrulama kaybı düşmeyi bırakınca eğitimi durdurmak gerekir.
Ezberleyen bir model yeni cümlelerde zorlanır.
Kendi sesimle konuşan bir model yapmak istedim.
Bu cümleyi söyleyen ben değilim, benim eğittiğim model.
Freya ekibinin Türkçe test cümleleriyle modelimi ölçtüm.
Demo Day için iki dakikalık bir konuşma hazırladım.
Başlarken türevin ne işe yaradığını bile tam bilmiyordum.
En çok öğrendiğim an, kaybın neden patladığını bulduğum gündü.
Şimdi bir modeli veriden ölçüme kadar kendim kurabiliyorum.
Sırada daha iyi bir veri seti ve daha temiz bir ses var.
Öğrenme oranı çok büyük olursa eğitim dağılır.
Küçük bir hata bütün eğitimi boşa çıkarabilir.
Grafik çizmeden önce sayılara bakmayı öğrendim.
Her hafta kodu yazdım, videoyu çektim ve teslim ettim.
Sinir ağları sihir değil, sadece çok sayıda çarpma ve toplama.
Eğitim verisi ne kadar temizse model o kadar iyi konuşur.
Yapay zekâ mühendisliği benim için yeni bir başlangıç.
Bu proje okul, iş ve ders arasında üç günde bitti.
Modelin ilk sesleri sadece gürültüydü.
Birkaç bin adım sonra mırıltı kelimelere dönüşmeye başladı.
Şimdi model cümleyi bitirince kendisi duruyor.
Dinlediğiniz için teşekkür ederim.
Sorularınız varsa memnuniyetle cevaplarım.
Kod ve sonuçlar GitHub sayfamda açık.
Bir sonraki adım bu modeli daha az veriyle daha iyi yapmak.""".split('\n')

def key(s): return re.sub(r"[^\w ]", "", norm_text(s)).strip()
def grams(s, n=5):
    w = key(s).split(); return {' '.join(w[k:k + n]) for k in range(len(w) - n + 1)}

# Freya-TR-Eval
fpath = os.path.join(HERE, 'freya_tr_eval.jsonl')
if not os.path.exists(fpath):
    subprocess.run(['curl', '-sfL', '-o', fpath, 'https://huggingface.co/datasets/freyavoice/freya-tr-eval/resolve/main/freya_tr_eval.jsonl'], check=True)
F = [key(json.loads(l)['text']) for l in open(fpath, encoding='utf-8')]
FG = set().union(*[grams(f) for f in F])
def clashes(s):
    k = key(s)
    return any(k == f or k in f or f in k for f in F) or bool(grams(s) & FG)

def ok(t):
    if re.search(r'\d', t) or not 25 <= len(t) <= 120: return False
    w = t.split()
    if not 4 <= len(w) <= 18: return False
    low = unicodedata.normalize('NFC', t).replace('I', 'ı').replace('İ', 'i').lower().translate(_MAP)
    if any(c not in stoi and c not in '"()' for c in low): return False             # okunamayan karakter yok
    if sum(1 for x in w[1:] if x[:1].isupper()) > 1: return False                     # en çok bir özel isim
    if re.search(r'[A-ZÇĞİÖŞÜ]{2,}', t): return False                                 # kısaltma yok
    return t.strip()[-1] in '.?!'

cand = json.load(open(os.path.join(HERE, 'alania_candidates.json'), encoding='utf-8'))
seen, pool, drop = set(), [], collections.Counter()
for c in cand:
    t = ' '.join(c['text'].split())
    if t[:1].islower(): t = {'i': 'İ', 'ı': 'I'}.get(t[0], t[0].upper()) + t[1:]      # Türkçe büyük harf
    if c['cat'] in ('foreign', 'abbrev', 'paragraph', 'long_explanations'): drop['kategori'] += 1; continue
    if not ok(t): drop['biçim'] += 1; continue
    if key(t) in seen: drop['tekrar'] += 1; continue
    if c['src'] != 'domain' and not (t[0].isupper() and len(t) <= 100 and '–' not in t): drop['web metni (sıkı süzgeç)'] += 1; continue
    if clashes(t): drop['freya örtüşmesi'] += 1; continue
    seen.add(key(t)); pool.append(dict(text=t, src='alania', cat=c['cat'], web=c['src'] != 'domain'))
own = [dict(text=t, src='own', cat='demo') for t in OWN if t.strip()]
assert not any(clashes(o['text']) for o in own), [o['text'] for o in own if clashes(o['text'])]
print('aday', len(cand), '| kullanılabilir', len(pool), '| atılan', dict(drop))

# Sıra: önce ses çeşitliliği (harf ikilisi kapsamı en çok artan cümle), sonra rastgele; kendi cümlelerim ilk 300'e serpiştirilir
def bigr(t): k = key(t); return {k[j:j + 2] for j in range(len(k) - 1)}
random.shuffle(pool)
need = N_TOTAL - len(own)
dom, web = [p for p in pool if not p['web']], [p for p in pool if p['web']]
rest = dom[:round(need * 0.7)] + web[:need - round(need * 0.7)]     # %70 alan metni, %30 web metni
random.shuffle(rest)
chosen, cov = [], set()
for _ in range(150):                                                 # kapsamı karakter başına en çok artıran
    b = max(rest, key=lambda p: len(bigr(p['text']) - cov) / len(p['text'])); rest.remove(b); chosen.append(b); cov |= bigr(b['text'])
chosen += rest
step = N_MIN // len(own)
for k, o in enumerate(own): chosen.insert(k * step, o)
chosen = chosen[:N_TOTAL]
out = [dict(id=f'r{k + 1:04d}', text=c['text'], src=c['src'], cat=c['cat']) for k, c in enumerate(chosen)]
json.dump(out, open(os.path.join(HERE, 'sentences.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=0)

chars = [len(norm_text(s['text'])) for s in out]
cps = 13.5                                                   # tahmini okuma hızı (karakter/sn); kayıtlar gelince ölçülür
print(f"{len(out)} cümle ({sum(s['src'] == 'own' for s in out)} kendi) | ortalama {sum(chars) / len(chars):.0f} karakter")
print(f"tahmini ses: ilk {N_MIN} = {sum(chars[:N_MIN]) / cps / 60:.0f} dk, hepsi = {sum(chars) / cps / 60:.0f} dk  (okuma hızı {cps} karakter/sn varsayımıyla)")
print('harf ikilisi kapsamı:', len(cov), '| Freya kontrolü: örtüşen', sum(clashes(s['text']) for s in out))
for s in out[:14] + out[300:304]: print(' ', s['id'], s['src'], '|', s['text'])
print('web oranı:', sum(1 for c in chosen if c.get('web')) / len(chosen))
