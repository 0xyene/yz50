# %%
# Veri: her örnek bölünmeden, BLOCK+1 uzunluğunda satırlara paketlenir. Boş kalan yer PAD.
def load_tokens(paths):
    toks, lens = [], []
    for p in paths:
        d = np.load(p); toks.append(d['tokens']); lens.append(d['lens'])
    return np.concatenate(toks), np.concatenate(lens)

def pack_rows(tokens, lens, seed):
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    rows, row, fill = [], np.zeros(BLOCK + 1, np.uint16), 0
    for i in np.random.default_rng(seed).permutation(len(lens)):
        n = int(lens[i])
        if n > BLOCK + 1: continue
        if fill + n > BLOCK + 1:
            rows.append(row); row, fill = np.zeros(BLOCK + 1, np.uint16), 0
        row[fill:fill + n] = tokens[starts[i]:starts[i] + n]; fill += n
    rows.append(row)
    return np.stack(rows)

def get_batch(rows, i):
    xb = torch.from_numpy(rows[i:i + BATCH].astype(np.int64)).to(DEVICE)
    x, y = xb[:, :-1], xb[:, 1:].clone()
    y[(y < AUDIO_BASE) & (y != EOS)] = -100          # loss yalnız ses token'ları ve EOS üstünde
    return x, y

train_tok, train_len = load_tokens(sorted(glob.glob(f'{TOK_DIR}/train_*.npz'))[:MAX_FILES])
for d, rep in EXTRA:                                 # ek veri: (dizin, kaç kez tekrar)
    t, l = load_tokens(sorted(glob.glob(f'{d}/train_*.npz')))
    train_tok, train_len = np.concatenate([train_tok] + [t] * rep), np.concatenate([train_len] + [l] * rep)
val_tok, val_len = load_tokens([VAL_PATH])
val_rows = pack_rows(val_tok, val_len, seed=0)
print(f'train: {len(train_len):,} örnek, {len(train_tok):,} token (~{(len(train_tok) - train_len.sum() * 0.12) / TOK_PER_S / 3600:.0f} saat ses) | val: {len(val_len):,} örnek, {len(val_rows)} satır')

