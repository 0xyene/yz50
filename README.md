# YZ50

My weekly assignment work for **[YZ50](https://yz50.ai/)** — a 12-week intensive program that takes participants in Turkey and trains them to build AI models from scratch rather than call them through an API.

The curriculum follows Andrej Karpathy's *Zero to Hero* path: a single neuron written without libraries, then autograd, PyTorch, character-level language models, embeddings and backpropagation by hand, ending in a small GPT built from the ground up. Mentored by researchers from Freya (YC S25).The program's stated goal is producing AI models in Turkey, not just users of them.


| Week | Topic | Files | Video (Turkish) |
|---|---|---|---|
| 1 | Single neuron → layer → loss → manual parameter search → numerical gradient descent | [`w1/`](w1/) | [▶ watch](https://youtu.be/WTg14e3sVxQ) |
| 2 | Backpropagation: a small micrograd — `Value` autograd engine, `backward()`, and an MLP trained with gradient descent | [`w2/`](w2/) | [▶ watch](https://youtu.be/6lc6fQNhuEE) |
| 3 | Bigram character language model, built by counting and again as a single-layer network trained with gradient descent; negative log likelihood, smoothing, and the same two models rerun on Turkish names | [`w3/`](w3/) | [▶ watch](https://youtu.be/mOg75QSEMgs) |
| 4 | MLP language model over a three-character context: learned embeddings, minibatch training and a learning rate sweep, then a look inside the network — high initial loss, tanh saturation, Kaiming init and BatchNorm — and the same model on Turkish names | [`w4/`](w4/) | [▶ watch](https://youtu.be/OmbdkcDRERk) |
| 5 | Backpropagation by hand: the week 4 model written as single operations, the gradient of every intermediate derived manually and checked against autograd | [`w5/`](w5/) | [▶ watch](https://youtu.be/3EEH55u7EFo) |
| 6 | WaveNet: the week 4 model rebuilt from layer classes, context 3 → 8, characters fused two at a time in a three-level tree, a BatchNorm bug on 3D input found and fixed, and the model on Turkish names | [`w6/`](w6/) | [▶ watch](https://youtu.be/lXYEbUDKSLs) |
| 7 | GPT from scratch, part 1: a character tokenizer and bigram baseline on Tiny Shakespeare, averaging over the past with a lower-triangular matrix and softmax, and a single self-attention head with positional embeddings and √head_size scaling, trained inside the language model | [`w7/`](w7/) | [▶ watch](https://youtu.be/d3T2BO2Sm4k) |

Each week's folder has its own README with the assignment list. Each week also has a walkthrough video, in Turkish, explaining the code.
