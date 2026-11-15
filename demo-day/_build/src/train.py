# %%
# Eğitim
rows = pack_rows(train_tok, train_len, seed=1000 + epoch)
print(f'bir tur = {len(rows) // BATCH:,} adım')
if step == 0: evaluate(0, float('nan'), 0.0)                     # eğitimsiz model: gürültü
raw.train()
t0, s0, recent = time.time(), step, []
while step < MAX_STEPS:
    if ptr + BATCH > len(rows):
        epoch += 1; ptr = 0; rows = pack_rows(train_tok, train_len, seed=1000 + epoch)
    x, y = get_batch(rows, ptr); ptr += BATCH
    lr = get_lr(step)
    for g in opt.param_groups: g['lr'] = lr
    with autocast(): _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
    opt.step()
    step += 1
    recent.append(loss.item()); recent = recent[-50:]
    tok_s = (step - s0) * BATCH * BLOCK / (time.time() - t0)
    if step % 50 == 0:
        left = (MAX_STEPS - step) * BATCH * BLOCK / tok_s / 60
        print(f'adım {step:6d} | loss {np.mean(recent):.4f} | lr {lr:.2e} | norm {norm:.2f} | {tok_s:,.0f} tok/sn | tur {epoch} | kalan ~{left:.0f} dk')
    if step % EVAL_EVERY == 0 or step == MAX_STEPS:
        evaluate(step, float(np.mean(recent)), tok_s)
    if step % CKPT_EVERY == 0 or step == MAX_STEPS:
        save_ckpt()
    if globals().get('SAVE_EVERY') and step % SAVE_EVERY == 0:        # 04c: her N adımda ayrı, küçük (bf16) checkpoint
        torch.save(dict(model={k: v.to(torch.bfloat16) for k, v in raw.state_dict().items()}, step=step, cfg=cfg),
                   f'{ROOT}/ckpt/{RUN}_step{step:05d}.pt')
print('bitti. checkpoint:', CKPT)

# %%
# İstediğin cümleyi söylet
a = speak('Bu modeli sıfırdan ben eğittim.', seed=1)
save_wav(f'{ROOT}/samples/{RUN}/{RUN}_deneme.wav', a)
try:
    from IPython.display import Audio, display
    display(Audio(a, rate=SR))
except Exception: pass
