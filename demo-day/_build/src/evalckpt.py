# %%
# Ölçüm: val loss + sabit cümlelerin sesi + log. Checkpoint.
@torch.no_grad()
def evaluate(step, train_loss, tok_s):
    raw.eval()
    nb = max(1, min(VAL_BATCHES, len(val_rows) // BATCH))
    vl = 0.0
    for i in range(nb):
        x, y = get_batch(val_rows, i * BATCH)
        with autocast(): _, loss = model(x, y)
        vl += loss.item() / nb
    files = []
    for k, text in enumerate(SAMPLE_TEXTS):
        a = speak(text)
        name = f'{RUN}_step{step:06d}_{k}.wav'; save_wav(f'{ROOT}/samples/{RUN}/{name}', a); files.append(name)
        try:
            from IPython.display import Audio, display
            display(Audio(a, rate=SR))
        except Exception: pass
    raw.train()
    prev_best = min([l['val'] for l in log], default=float('inf'))
    log.append(dict(step=step, train=train_loss, val=vl, lr=get_lr(step), tok_s=tok_s, samples=files))
    json.dump(dict(run=RUN, cfg=cfg, texts=SAMPLE_TEXTS, log=log), open(LOG, 'w'), ensure_ascii=False)
    if vl < prev_best:                                   # en iyi val loss'lu ağırlıklar ayrıca saklanır
        torch.save(dict(model=raw.state_dict(), step=step, val=vl, cfg=cfg), BEST + '.part'); os.replace(BEST + '.part', BEST)
    print(f'--- adım {step}: val loss {vl:.4f}{"  (en iyi, kaydedildi)" if vl < prev_best else ""} | örnekler: {files}')

def save_ckpt():
    torch.save(dict(model=raw.state_dict(), opt=opt.state_dict(), step=step, epoch=epoch, ptr=ptr, log=log, cfg=cfg), CKPT + '.part')
    os.replace(CKPT + '.part', CKPT)

