# YZ50 — Week 3

Assignment solutions for YZ50.

The first language model of the course: a bigram character model, built twice.
Once by counting, once by training a single-layer network with gradient descent —
and the two land on the same number.

| # | Task | File |
|---|---|---|
| 1 | Count bigrams over `names.txt`, first with a Python dictionary and then in a 27×27 tensor. Cross-check the two totals against each other, and visualise the table. | `yz50_w3_1.ipynb` |
| 2 | Turn the counts into per-row probability distributions and sample new names. Includes the `keepdim` trap run side by side with the correct version, to show that the broken model produces plausible-looking samples. | `yz50_w3_2.ipynb` |
| 3 | Negative log likelihood as a single measure of model quality, plus `+1` smoothing so that unseen bigrams do not send the loss to infinity. | `yz50_w3_3.ipynb` |
| 4 | The same model as a neural network: one-hot input, a 27×27 weight matrix, softmax, NLL loss, and a gradient descent loop. The loss curve is plotted against the count model's closed-form value. | `yz50_w3_4.ipynb` |
| 5 | Both models rerun on a Turkish name list, with the alphabet extended to ç ğ ı ö ş ü. Loss, loss curve, and sampled names from each model. | `yz50_w3_5.ipynb` |

Task 6 (trigram with a train/dev/test split) was optional and is not included.

## Data

| File | Contents |
|---|---|
| `names.txt` | 32,033 English names, from Karpathy's `makemore` repository. |
| `isimler_unique.txt` | 9,721 Turkish names, collected and cleaned for task 5: lowercased, deduplicated, no empty lines, restricted to the 29 letters of the Turkish alphabet. |

## Results

Mean negative log likelihood, lower is better.

| Data | Count model | Count model, `+1` | Neural net, 101 steps |
|---|---|---|---|
| English, 32,033 names, vocab 27 | 2.4540 | 2.4546 | 2.4727 |
| Turkish, 9,721 names, vocab 30 | 2.4762 | 2.4794 | 2.4997 |

The count table is the closed-form maximum likelihood solution, so gradient descent
approaches it from above and cannot beat it. With more steps the gap keeps closing —
on the English data, to +0.0006 by 2,000 steps.

Sampled Turkish names, same seed, both models:

```
güsu.  oçmerslbi.  te.  ören.  yhimur.
```

Both routes produce these character for character, which is the clearest statement
of the week's point: counting and gradient descent are two ways to the same model.
