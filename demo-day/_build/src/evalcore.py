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

