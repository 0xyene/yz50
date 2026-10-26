# YZ50 — Week 9

Assignment solutions for YZ50.

Building a GPT tokenizer from scratch, following Karpathy's "Let's build the GPT
Tokenizer". Text is turned into UTF-8 bytes, a byte pair encoding (BPE) tokenizer is
trained on a Turkish corpus, and `encode` / `decode` are written on top of it. The
tokenizer is then retrained after splitting the text with GPT-4's regex pattern, and
token counts for the same paragraph are compared between Turkish and English.

| # | Task | File |
|---|---|---|
| 1 | A Turkish sentence as Unicode code points and as UTF-8 bytes. Its 35 characters become 43 bytes because each of the 8 Turkish letters takes 2 bytes. | `yz50_w9_1.ipynb` |
| 2 | BPE written from scratch and trained on the Turkish corpus with 50 merges (vocabulary size 306). The merges are printed as text. Compression ratio: 1.52 bytes per token. | `yz50_w9_2.ipynb` |
| 3 | `encode` and `decode` for the trained tokenizer. A string with an emoji and the whole corpus both come back unchanged from `decode(encode(text))`. | `yz50_w9_3.ipynb` |
| 4 | The corpus is split into chunks with GPT-4's regex pattern, and BPE is retrained with 50 merges, counting pairs only inside each chunk. The merges are printed as text. `Ankara'dan` splits into `Ankara`, `'d`, `an`; with the curly apostrophe, `Ankara’dan` splits into `Ankara`, `’dan`. The same two sentences are also split in Turkish and in English. | `yz50_w9_4.ipynb` |
| 5 | The same paragraph in Turkish (49 words) and English (88 words), tokenized with GPT-4's tokenizer (tiktoken, `cl100k_base`) and with the tokenizer from task 2. Tokens per word: 3.35 (Turkish) and 1.08 (English) with GPT-4; 6.24 and 4.53 with the own tokenizer. The word `kullanabilecekken` is shown as pieces for both tokenizers. | `yz50_w9_5.ipynb` |

## Data

| File | Contents |
|---|---|
| `altayli_kose_yazilari.txt` | The Turkish corpus from week 8: 218 newspaper columns by Fatih Altaylı, 1,007,987 characters (1,117,957 bytes in UTF-8). Not included in the repository because the columns are under copyright. |
