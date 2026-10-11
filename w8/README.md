# YZ50 — Week 8

Walkthrough video (in Turkish): https://youtu.be/Gf4T_sxAcVE

Assignment solutions for YZ50.

Part 2 of building a GPT from scratch. The single head from week 7 becomes multi-head
attention, and the Transformer block is completed one piece at a time (feedforward,
residual connections, LayerNorm) with the val loss recorded after each step. The model
is then scaled up with dropout, trained on a GPU, and trained again on a Turkish corpus.

| # | Task | File |
|---|---|---|
| 1 | The single head split into 4 heads of size 8 whose outputs are concatenated. The attention layer keeps the same 3,072 parameters as one head of size 32. | `yz50_w8_1.ipynb` |
| 2a | A feedforward layer (`Linear` + `ReLU`) after attention. | `yz50_w8_2a_ff.ipynb` |
| 2b | Three Transformer blocks with residual connections, a projection after attention, and a feedforward layer 4× wider inside. | `yz50_w8_2b_residual.ipynb` |
| 2c | Pre-norm LayerNorm before attention and feedforward in each block, plus a final LayerNorm before the output layer. | `yz50_w8_2c_layernorm.ipynb` |
| 3 | The model scaled up (`n_embd` 384, 6 heads, 6 layers, `block_size` 256) with dropout 0.2, trained on a Colab A100. Parameter count, training time, val loss and a text sample are printed. | `yz50_w8_3.ipynb` |
| 4 | The same model trained on a Turkish corpus of about 1 MB, with the number of distinct characters compared to Tiny Shakespeare and a sample generated from the prompt `NE ZAMAN ADAM OLURUZ?`. | `yz50_w8_4.ipynb` |

Task 5 was optional and is not included.

## Data

| File | Contents |
|---|---|
| `input.txt` | Tiny Shakespeare, 1,115,394 characters, 65 distinct, from Karpathy's `char-rnn` repository. |
| `altayli_kose_yazilari.txt` | 218 newspaper columns by Fatih Altaylı, mostly from 2007–2008, collected from [fatihaltayli.com.tr](https://fatihaltayli.com.tr/yazilar/fatih-altayli/kose-yazilari). 1,007,987 characters (1,117,957 bytes in UTF-8), 103 distinct. Not included in the repository because the columns are under copyright. |

## Results

### Task 1 and 2: the block piece by piece

Tiny Shakespeare, same split, `n_embd` 32, `block_size` 8, batch size 32, 200,000 steps,
AdamW with lr 1e-3, identical seed. Val loss is the mean over 200 random batches. The
first two rows are from week 7.

| Model | Params | val |
|---|---|---|
| Bigram | 4,225 | 2.491 |
| One self-attention head | 7,553 | 2.345 |
| Multi-head attention, 4 × 8 (task 1) | 7,553 | 2.159 |
| + feedforward (task 2a) | 8,609 | 2.105 |
| + residual, 3 blocks (task 2b) | 41,921 | 1.904 |
| + LayerNorm (task 2c) | 42,369 | 1.898 |

### Task 3 and 4: the scaled-up model

`n_embd` 384, 6 heads, 6 layers, `block_size` 256, dropout 0.2, batch size 64,
5,000 steps, AdamW with lr 3e-4, Colab A100. Train loss is the last batch.

| Data | Distinct characters | Params | train | val | Training time |
|---|---|---|---|---|---|
| Tiny Shakespeare (task 3) | 65 | 10,788,929 | 1.165 | 1.504 | 9.4 min |
| Fatih Altaylı columns (task 4) | 103 | 10,818,151 | 1.103 | 1.312 | 9.4 min |
