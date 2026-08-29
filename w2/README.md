# YZ50 — Week 2

Assignment solutions for YZ50.

| # | Task | File |
|---|---|---|
| 1 | Write your own `Value` class, starting with addition and multiplication. Each new `Value` stores the Values that produced it and which operation it came from. | `yz50_w2_1.py` |
| 2 | Before automating anything, fill in the gradients by hand on the two examples from the video: first the simple expression, then a single neuron. `tanh` is added to `Value` here. | `yz50_w2_2.ipynb` |
| 3 | Write `backward()`: set the output gradient to 1, walk the nodes in reverse topological order, apply the chain rule at each node, accumulating gradients rather than overwriting them. | `yz50_w2_3.ipynb` |
| 4 | Decompose `tanh` into `exp`, division and `pow`, and show the gradients are unchanged. Then verify the same expression three ways: `backward()`, the numerical derivative, and PyTorch. | `yz50_w2_4.ipynb` |
| 5 | Build `Neuron`, `Layer` and `MLP`, collect the parameters in a single list, train on the small dataset from the video, and show the loss dropping step by step. | `yz50_w2_5.ipynb` |
