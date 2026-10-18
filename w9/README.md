# YZ50 — Week 9

Assignment solutions for YZ50.

Building a GPT tokenizer from scratch, following Karpathy's "Let's build the GPT
Tokenizer". Text is turned into UTF-8 bytes, a byte pair encoding (BPE) tokenizer is
trained on a Turkish corpus, and `encode` / `decode` are written on top of it.

| # | Task | File |
|---|---|---|
| 1 | A Turkish sentence as Unicode code points and as UTF-8 bytes. Its 35 characters become 43 bytes because each of the 8 Turkish letters takes 2 bytes. | `yz50_w9_1.ipynb` |
| 2 | BPE written from scratch and trained on the Turkish corpus with 50 merges (vocabulary size 306). The merges are printed as text. Compression ratio: 1.52 bytes per token. | `yz50_w9_2.ipynb` |
| 3 | `encode` and `decode` for the trained tokenizer. A string with an emoji and the whole corpus both come back unchanged from `decode(encode(text))`. | `yz50_w9_3.ipynb` |

Tasks 4 and 5 are not included yet.

## Data

| File | Contents |
|---|---|
| `altayli_kose_yazilari.txt` | The Turkish corpus from week 8: 218 newspaper columns by Fatih Altaylı, 1,007,987 characters (1,117,957 bytes in UTF-8). Not included in the repository because the columns are under copyright. |
