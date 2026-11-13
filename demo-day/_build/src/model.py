# %%
# Model: GPT-2 mimarisi, sıfırdan
class CausalSelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_attn = nn.Linear(c['n_embd'], 3 * c['n_embd'])
        self.c_proj = nn.Linear(c['n_embd'], c['n_embd']); self.c_proj.RESID = True
        self.n_head, self.dropout = c['n_head'], c['dropout']
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = [t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_fc = nn.Linear(c['n_embd'], 4 * c['n_embd'])
        self.c_proj = nn.Linear(4 * c['n_embd'], c['n_embd']); self.c_proj.RESID = True
        self.drop = nn.Dropout(c['dropout'])
    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x), approximate='tanh')))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln_1, self.attn = nn.LayerNorm(c['n_embd']), CausalSelfAttention(c)
        self.ln_2, self.mlp = nn.LayerNorm(c['n_embd']), MLP(c)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))

class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c['vocab'], c['n_embd'])
        self.wpe = nn.Embedding(c['block'], c['n_embd'])
        self.h = nn.ModuleList([Block(c) for _ in range(c['n_layer'])])
        self.ln_f = nn.LayerNorm(c['n_embd'])
        self.lm_head = nn.Linear(c['n_embd'], c['vocab'], bias=False)
        self.wte.weight = self.lm_head.weight                      # weight tying
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02 * (2 * self.c['n_layer']) ** -0.5 if hasattr(m, 'RESID') else 0.02
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None, last_only=False):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.h: x = b(x)
        x = self.ln_f(x)
        if last_only: return self.lm_head(x[:, -1, :]), None
        logits = self.lm_head(x)
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

cfg = dict(vocab=VOCAB, block=BLOCK, n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, dropout=DROPOUT)

