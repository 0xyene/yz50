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

Task 3 (Karpathy's exercises 2–4 — cross entropy and BatchNorm each in a single
expression, then training without `loss.backward()`) was optional and is not included.

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

## Two mistakes worth keeping

Neither of these raised an error. Both were found by reading the number `cmp` prints.

`dlogits` came out with `maxdiff: 1.0`. The second path into `logits` runs through
`logits.max(1)`, and `max` sends the gradient to the largest element of each row — not
to the target character. Scattering it at `Yb`'s positions put a 1.0 in the wrong 32
places.

`dbndiff` came out with `maxdiff: 9.4e-4` while `dhprebn`, computed from it, was exact.
`dhprebn = dbndiff` names the same tensor instead of copying it, so the `+=` on the
following line also landed inside `dbndiff`. The size of the difference matched
`(1/n)·dbnmeani`, which is what pointed at the line; `.clone()` fixes it.

One more thing `cmp` does not catch: it compares values, not shapes. A `(64,)` gradient
for a `(1, 64)` parameter broadcasts and still reports `exact`, so shapes are worth
checking separately.

The embedding gradient is an explicit loop over the 96 context positions. `index_add_`
does the same in one call — the loop is kept here because it shows why the duplicate
indices have to accumulate rather than overwrite.
