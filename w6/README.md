# YZ50 — Week 6

Walkthrough video (in Turkish): https://youtu.be/lXYEbUDKSLs

Assignment solutions for YZ50.

The week 4 model is rebuilt from layer classes, in the style of `torch.nn`, and the
context grows from 3 to 8 characters. The flat MLP that concatenates all 8 characters
at once is then replaced by a WaveNet-style tree: characters are fused two at a time,
8 → 4 → 2 → 1, over three layers. Printing the shapes along the way exposes a bug
in BatchNorm on 3D input.

| # | Task | File |
|---|---|---|
| 1 | `Linear`, `BatchNorm1d`, `Tanh`, `Embedding`, `Flatten` and `Sequential` classes; the model is a single `Sequential` and the training loop only calls `model(Xb)` and `model.parameters()`. The loss curve is averaged over 1,000-step windows. | `yz50_w6_1.ipynb` |
| 2 | Context 3 → 8 with nothing else changed: the baseline for the rest of the week. | `yz50_w6_2.ipynb` |
| 3 | WaveNet: `FlattenConsecutive(2)` before each of three hidden layers, with the output shape of every layer printed. | `yz50_w6_3.ipynb` |
| 4 | BatchNorm1d on 3D input took its statistics over dim 0 only, so each of the 4 positions got its own mean: `running_mean` is `(1, 4, 68)`. Averaging over dims `(0, 1)` gives one mean per channel, `(1, 1, 68)`. Retrained with the fix. | `yz50_w6_4.ipynb` |
| 5 | Scaled-up WaveNet (`n_embd` 24, `n_hidden` 128) and the comparison table. | `yz50_w6_5.ipynb` |
| 6 | The task 5 WaveNet on the Turkish name list, compared with the week 4 Turkish MLP. | `yz50_w6_6.ipynb` |

Task 3 keeps the BatchNorm bug on purpose: tasks 3 and 4 are the same run before and
after the fix. Task 7 was optional and is not included.

## Data

| File | Contents |
|---|---|
| `names.txt` | 32,033 English names, from Karpathy's `makemore` repository. |
| `isimler_unique.txt` | 9,721 Turkish names, the same list and the same split as week 4. |

## Results

Dev loss, lower is better. English data, 200,000 steps, identical seed and split
throughout.

| Model | Context | Params | train | dev |
|---|---|---|---|---|
| MLP (task 1) | 3 | 12,097 | 2.058 | 2.107 |
| Flat MLP (task 2) | 8 | 22,097 | 1.916 | 2.034 |
| WaveNet, BatchNorm bug (task 3) | 8 | 22,397 | 1.945 | 2.032 |
| WaveNet, BatchNorm fixed (task 4) | 8 | 22,397 | 1.911 | 2.020 |
| WaveNet, scaled up (task 5) | 8 | 76,579 | 1.769 | 1.994 |

Turkish names:

| Model | Context | Params | train | dev |
|---|---|---|---|---|
| Week 4 MLP | 3 | 12,730 | 1.881 | 2.077 |
| WaveNet (task 6) | 8 | 77,038 | 1.353 | 2.282 |

The Turkish training set is only 55,346 examples. The 77K-parameter WaveNet fits it
far better than the week 4 MLP but does worse on the dev set.
