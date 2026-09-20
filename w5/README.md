# YZ50 — Week 5

Assignment solutions for YZ50.

No new model this week. The MLP + BatchNorm from week 4 is written out as a chain of
single operations, and the gradient of every intermediate in that chain is derived by
hand. `loss.backward()` is used only to produce the answer key to check against.

| # | Task | File |
|---|---|---|
| 1 | Break the forward pass into one-operation steps (`logit_maxes`, `norm_logits`, `counts`, `counts_sum`, `probs`, `logprobs`, …), keep every intermediate gradient with `retain_grad()` and run `loss.backward()`. | `yz50_w5_1_2.ipynb` |
| 2 | Derive the same gradients by hand, one node at a time, and compare each one with `cmp`. | `yz50_w5_1_2.ipynb` |

Both tasks are in one notebook: task 1 builds the forward pass that task 2 differentiates.

## Data

| File | Contents |
|---|---|
| `names.txt` | 32,033 English names, from Karpathy's `makemore` repository. |

## Result

One minibatch of 32 examples, 10-dimensional embeddings, 64 hidden units. All 26
comparisons come out `exact`, with `maxdiff: 0.0` — identical bits, not merely close:

```
logprobs · probs · counts_sum_inv · counts_sum · counts · norm_logits · logit_maxes
logits · h · W2 · b2 · hpreact · bngain · bnbias · bnraw · bnvar_inv · bnvar
bndiff2 · bndiff · bnmeani · hprebn · embcat · W1 · b1 · emb · C
```

The embedding gradient is an explicit loop over the 96 context positions. `index_add_`
does the same in one call — the loop is kept here because it shows why the duplicate
indices have to accumulate rather than overwrite.
