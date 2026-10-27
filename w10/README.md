# YZ50 — Week 10

Assignment solutions for YZ50.

Part 1 of reproducing GPT-2 (124M), following the first hour of Karpathy's "Let's reproduce
GPT-2 (124M)". The pretrained weights from Hugging Face are inspected, then loaded into a
GPT-2 module written from the week 8 GPT. The module generates text with top-k sampling,
and a randomly initialized copy is checked before any real training.

| # | Task | File |
|---|---|---|
| 1 | The pretrained GPT-2 weights from Hugging Face: the name and shape of every tensor, the total of 124,439,808 parameters, and plots of the position embeddings (`wpe`) with the parameter count of each part of the model. | `yz50_w10_1.ipynb` |
| 2 | GPT-2 written as a PyTorch module from the week 8 GPT. The Hugging Face weights are loaded into it, with four weight matrices transposed, and its logits match the Hugging Face model's on the same input (`torch.allclose` is `True`). | `yz50_w10_2.ipynb` |
| 3 | A sampling loop with top-k (k = 50), five samples each for an English and a Turkish prompt. In the GPT-2 tokenizer the English prompt is 8 tokens and the Turkish prompt is 10. | `yz50_w10_3.ipynb` |
| 4 | A randomly initialized model starts at a loss of 11.03, close to ln(50257) ≈ 10.82. Trained on one batch of 4 × 32 tokens for 100 steps, the loss falls to 0.0015. | `yz50_w10_4.ipynb` |

Task 5 was optional and is not included.

## Data

| File | Contents |
|---|---|
| `input.txt` | Tiny Shakespeare, as in weeks 7 and 8. Task 4 uses its first 1,000 characters. |
