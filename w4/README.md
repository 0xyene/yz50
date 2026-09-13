# YZ50 — Week 4

Assignment solutions for YZ50.

The first real neural language model of the course. No table this time: every
character becomes a small learned vector, three of them are concatenated and fed
to an MLP, and the network predicts the next character — the model from Bengio's
2003 paper. The second half of the week looks inside the network while it trains:
why the initial loss is so high, why tanh saturates, what initialisation and
BatchNorm actually fix.

| # | Task | File |
|---|---|---|
| 1 | Build the dataset that takes the previous three characters as context (X: three indices, Y: the next character), create the 27×2 embedding table, and pull the embeddings out by indexing. | `yz50_w4_1.ipynb` |
| 2 | Hidden and output layers: flatten the embeddings, `W1` and `b1` through `tanh`, `W2` and `b2` into logits. The loss is computed by hand first, then shown to match `F.cross_entropy`. | `yz50_w4_2.ipynb` |
| 3 | The training loop: overfit a single minibatch first, then train on the full data with minibatches. Learning rate swept the way the video does it, and the data split into train / dev / test. | `yz50_w4_3.ipynb` |
| 4 | Scale up the hidden layer and the embedding and show what happens to the dev loss. The embedding is plotted in 2D, names are sampled from the model and put next to the bigram model's. | `yz50_w4_4.ipynb` |
| 5 | Why the initial loss is far above chance and why tanh saturates, with the histograms from the video. Kaiming init scales the weights and both problems go away. | `yz50_w4_5.ipynb` |
| 6 | A BatchNorm layer after the hidden layer: batch statistics during training, running estimates at prediction time. Dev loss compared with and without it. | `yz50_w4_6.ipynb` |
| 7 | The same model trained on last week's Turkish name list, with sampled Turkish names next to the bigram model's. | `yz50_w4_7.ipynb` |

Task 8 (Karpathy's follow-up exercises) was optional and is not included.

## Data

| File | Contents |
|---|---|
| `names.txt` | 32,033 English names, from Karpathy's `makemore` repository. |
| `isimler_unique.txt` | 9,721 Turkish names, the same list used in week 3. |

## Results

Dev loss, lower is better. English data, 10-dimensional embedding, 200 hidden
units, 200,000 steps, identical seed and split throughout — only the
initialisation changes.

| Initialisation | train | dev |
|---|---|---|
| Plain `randn`, nothing scaled | 2.1268 | 2.1698 |
| Output layer scaled (`W2` ×0.01, `b2` ×0) | 2.0696 | 2.1311 |
| Output and hidden scaled (`W1` ×0.15 by hand) | 2.0405 | 2.1018 |
| Kaiming, `(5/3)/√30` | 2.0377 | 2.1070 |
| BatchNorm with running statistics | 2.0674 | 2.1057 |

The first row starts at a loss of 27.9 on step 0, where a model that knew nothing
at all would sit at `-log(1/27) = 3.30`. Scaling the output layer alone brings
step 0 down to 3.32 and takes 0.039 off the dev loss — the largest single gain of
the week, from two multiplications.

Kaiming's `(5/3)/√30 ≈ 0.304` and the 0.15 chosen by hand land within 0.005 of
each other despite differing by a factor of two, which is the point: the order of
magnitude is what matters, not the decimal.

BatchNorm changes the dev loss by 0.0013 — nothing. Its value here is not a better
number but that `bngain = 1`, `bnbias = 0` and no tuning at all reaches the same
place the hand-scaled initialisations needed searching to find.

## Turkish

| Model | Loss |
|---|---|
| Bigram, by counting (week 3) | 2.4762 |
| MLP, dev | 2.0770 |

The bigram figure is over the whole list and the MLP figure is a dev split, so the
two are not strictly the same measurement, but the gap is far larger than that
distinction.

Sampled names, bigram above, MLP below:

```
güsu.     oçmerslbi.  te.      ören.    yhimur.
merzık.   akan.       nazat.   serik.   savet.   ülkutlu.
```

Three characters of memory instead of one is the whole difference. The MLP still
produces `ğahan.` — no Turkish word begins with ğ, and a three-character window
never sees enough of a word's start to learn that.
