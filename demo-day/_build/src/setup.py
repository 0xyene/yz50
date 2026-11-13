# %%
# Kurulum: model, optimizer, (varsa) checkpoint'ten devam
if DEVICE == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
os.makedirs(f'{ROOT}/ckpt', exist_ok=True); os.makedirs(f'{ROOT}/samples/{RUN}', exist_ok=True)
CKPT, LOG, BEST = f'{ROOT}/ckpt/{RUN}_last.pt', f'{ROOT}/{RUN}_log.json', f'{ROOT}/ckpt/{RUN}_best.pt'
torch.manual_seed(1337)
raw = GPT(cfg).to(DEVICE)
decay = [p for p in raw.parameters() if p.dim() >= 2]
no_decay = [p for p in raw.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{'params': decay, 'weight_decay': WEIGHT_DECAY}, {'params': no_decay, 'weight_decay': 0.0}],
                        lr=MAX_LR, betas=(0.9, 0.95), eps=1e-8, fused=(DEVICE == 'cuda'))
step, epoch, ptr, log = 0, 0, 0, []
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=DEVICE)
    raw.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
    step, epoch, ptr, log = ck['step'], ck['epoch'], ck['ptr'], ck['log']
    print(f'checkpoint bulundu: adım {step} ile devam')
elif INIT_FROM:
    raw.load_state_dict(torch.load(INIT_FROM, map_location=DEVICE)['model'])
    print('başlangıç ağırlıkları:', INIT_FROM)
model = torch.compile(raw) if COMPILE else raw
print(f'parametre: {sum(p.numel() for p in raw.parameters()):,}')

def get_lr(it):
    if it < WARMUP: return MAX_LR * (it + 1) / WARMUP
    if it >= MAX_STEPS: return MIN_LR
    r = (it - WARMUP) / (MAX_STEPS - WARMUP)
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * r)) * (MAX_LR - MIN_LR)

def autocast():
    return torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=AMP)

