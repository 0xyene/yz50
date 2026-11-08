# YZ50 — Week 12

Assignment solutions for YZ50.

Part 3 of reproducing GPT-2 (124M), following Karpathy's "Let's reproduce GPT-2 (124M)" from 3:10:21 to the end.
The model from week 11 is trained on real data: a part of FineWeb-Edu, split into token shards.
During training it measures validation loss and writes text samples.
At the end it is measured on HellaSwag and compared with OpenAI's GPT-2 weights.

| # | Task | File |
|---|---|---|
| 1 | 100M tokens of FineWeb-Edu (sample-10BT, streamed), tokenized with the GPT-2 tokenizer into 10 shards of 10M tokens. The first shard is the validation shard, the other 9 are for training (90M tokens). The data loader reads the shards one after another. | `yz50_w12_1.ipynb` |
| 2 | Validation loss (20 batches of 16 × 1024) every few steps, and 4 text samples with the same prompt and the same seed. Train and validation loss in one chart. Tested with a 20-step run. | `yz50_w12_2.ipynb` |
| 3 | 600 steps on an A100 with the week 11 setup (524,288 tokens per step, warmup 40 steps, cosine learning rate). The model saw 314,572,800 tokens (about 3.5 passes over the training shards) at 182,150 tokens per second, in 29 minutes. Final train loss 4.854, final validation loss 4.827. The samples go from random tokens (step 0) to single words (step 100) to fluent English without meaning (step 599). | `yz50_w12_3.ipynb` |
| 4 | HellaSwag on the first 1,000 validation examples (acc_norm). OpenAI GPT-2 (124M): 336 / 1000 = 33.6%. My model: 284 / 1000 = 28.4%. Random guessing gives 25%. My model's run is at the end of `yz50_w12_3.ipynb`, on the model that was just trained. | `yz50_w12_4.ipynb` |

Task 5 was optional and is not included.

## Data

The FineWeb-Edu shards (200 MB) and the model weights are on Google Drive, not in this repo.
The original HellaSwag download link (github.com/rowanz/hellaswag) does not work anymore, so the validation set was taken from Hugging Face (`Rowan/hellaswag`) as `hellaswag_val.jsonl`. It is not in this repo either.
