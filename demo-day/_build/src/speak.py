# %%
# Konuşturma: metni ver, ses token'larını tek tek örnekle, Mimi ile dalgaya çevir
mimi = MimiModel.from_pretrained('kyutai/mimi').to(DEVICE).eval()

@torch.no_grad()
def speak(text, tag=SAMPLE_TAG, seconds=SAMPLE_SECONDS, temp=TEMP, top_k=TOP_K, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.tensor([[BOS, tag] + encode_text(text) + [SEP]], device=DEVICE)
    out = []
    while len(out) < int(seconds * TOK_PER_S) and x.shape[1] < BLOCK:
        with autocast(): logits, _ = raw(x, last_only=True)
        logits = logits[0].float()
        cb = len(out) % N_CB                                   # sıradaki token hangi codebook'tan
        lo = AUDIO_BASE + cb * CB_SIZE
        mask = torch.full_like(logits, float('-inf')); mask[lo:lo + CB_SIZE] = 0
        if cb == 0 and len(out) >= N_CB: mask[EOS] = 0         # EOS yalnız kare sınırında
        v, ix = torch.topk((logits + mask) / temp, top_k)
        tok = ix[torch.multinomial(F.softmax(v, dim=-1), 1, generator=g)]
        if tok.item() == EOS: break
        out.append(tok.item() - lo)
        x = torch.cat([x, tok.view(1, 1)], dim=1)
    n = len(out) // N_CB
    if n < 2: return np.zeros(SR // 10, dtype=np.float32)
    codes = torch.tensor(out[:n * N_CB], device=DEVICE).view(n, N_CB).T[None]      # (1, 8, kare)
    return mimi.decode(codes).audio_values[0, 0].float().cpu().numpy()

def save_wav(path, a):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())

