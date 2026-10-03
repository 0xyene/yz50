# YZ50 — Week 7

Assignment solutions for YZ50.

Part 1 of building a GPT from scratch, on Tiny Shakespeare. A character tokenizer and
a bigram model set the baseline. Averaging over the past is then written three ways,
ending in the masked softmax that attention uses, and a single self-attention head with
positional embeddings is built, scaled, and trained inside the language model.

| # | Task | File |
|---|---|---|
| 1 | Character tokenizer (`encode`, `decode`), 90/10 train/val split, `get_batch` over random chunks, and a bigram `nn.Module` trained as the baseline. Val loss is the mean over 200 random batches. | `yz50_w7_1.ipynb` |
| 2 | Averaging over the past three ways: a for loop, a normalized `tril` matmul, and `masked_fill` with `-inf` followed by softmax. All three match under `torch.allclose`, as does a `triu` variant. Filling with `+inf` instead gives NaN, because softmax computes ∞ − ∞. | `yz50_w7_2.ipynb` |
| 3 | One self-attention head: query, key, value and positional embedding, with `wei` printed. Dividing the scores by √head_size: the variance of q·k drops from 15.76 to 0.98 (head_size 16), and softmax on the same q and k is shown unscaled and scaled. | `yz50_w7_3.ipynb` |
| 4 | The head inside the language model (token + positional embedding → head → linear), trained and compared with the task 1 baseline. Text is generated from the prompt `Romeo:`, with the context cropped to the last `block_size` characters. | `yz50_w7_4.ipynb` |

Task 5 was optional and is not included.

## Data

| File | Contents |
|---|---|
| `input.txt` | Tiny Shakespeare, 1,115,394 characters, from Karpathy's `char-rnn` repository. |

## Results

Val loss, lower is better. Same split, `block_size` 8, batch size 32, 200,000 steps,
AdamW with lr 1e-3, identical seed. Val loss is the mean over 200 random batches.

| Model | Params | val |
|---|---|---|
| Bigram (task 1) | 4,225 | 2.491 |
| Bigram + one self-attention head (task 4) | 7,553 | 2.345 |
