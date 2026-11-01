# YZ50 — Week 11

Assignment solutions for YZ50.

Part 2 of reproducing GPT-2 (124M), following Karpathy's "Let's reproduce GPT-2 (124M)" from 1:02:00 to 2:46:52.
The model from week 10 gets a data loader, weight tying and the GPT-2 initialization.
Training is then made faster one step at a time and measured after each step.
Last, the GPT-3 training settings are added: AdamW, gradient clipping, a learning rate schedule, weight decay and gradient accumulation.

| # | Task | File |
|---|---|---|
| 1 | A data loader over Tiny Shakespeare (338,025 tokens). The token embedding and the output layer share one weight matrix, which saves 38,597,376 parameters. Weights start at std 0.02. The two residual projections in each block start at 0.02 / √(2 · 12) ≈ 0.0041. The model has 124,439,808 parameters. | `yz50_w11_1.ipynb` |
| 2 | Step time and tokens per second, measured after each speed-up. Every run is one notebook, kept as it ran. TF32 gave the largest gain (2.74×). All five steps together make a step 11.1× faster. | see below |
| 3 | AdamW with betas (0.9, 0.95) and fused kernels. Gradients are clipped at norm 1.0, and the norm is printed every step. The learning rate warms up for 10 steps to 6e-4, then follows a cosine down to 6e-5. Weight decay 0.1 goes only to the 50 2D tensors (124,354,560 parameters), not to the 98 others (121,344). Gradient accumulation builds a batch of 524,288 tokens from 32 micro-batches of 16 × 1024. In 50 steps on an A100 the loss falls from 10.94 to 5.79, at 186,148 tokens per second. | `yz50_w11_3.ipynb` |

Task 4 was optional and is not included.

## Task 2 runs

GPT-2 124M, mean of steps 1–49 (step 0 is left out).

| File | Machine and batch | Change | ms / step | Tokens / s |
|---|---|---|---|---|
| `yz50_w11_2_m1.ipynb` | Apple M1 8 GB, 4 × 256 | float32 | 1,100 | 931 |
| `yz50_w11_2_a100.ipynb` | A100 40 GB, 4 × 256 | float32 | 65 | 15,761 |
| `yz50_w11_2a_float32.ipynb` | A100 40 GB, 16 × 1024 | float32 | 1,039 | 15,770 |
| `yz50_w11_2b_tf32.ipynb` | A100 40 GB, 16 × 1024 | + TF32 | 379 | 43,209 |
| `yz50_w11_2c_bf16.ipynb` | A100 40 GB, 16 × 1024 | + bfloat16 | 330 | 49,606 |
| `yz50_w11_2d_compile.ipynb` | A100 40 GB, 16 × 1024 | + torch.compile | 138 | 118,845 |
| `yz50_w11_2e_flash.ipynb` | A100 40 GB, 16 × 1024 | + flash attention | 97 | 168,925 |
| `yz50_w11_2f_vocab50304.ipynb` | A100 40 GB, 16 × 1024 | + vocab 50,304 | 93 | 175,436 |

The M1 has no TF32 and no native bfloat16, and 8 GB fits at most a 4 × 256 batch in float32.
So the speed-up steps were measured on an A100.

## Data

| File | Contents |
|---|---|
| `input.txt` | Tiny Shakespeare, as in weeks 7, 8 and 10. |
