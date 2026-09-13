#4. Parametreleri manuel değiştirerek loss'un nasıl değiştiğini gözlemle, loss eğrisini çiz. 
import numpy as np
import matplotlib.pyplot as plt

input  = [1,2,3,4]
y_true = [3,8,5,4]
bias    = -0.8
def loss(w):
    total_loss = 0 
    for i in range(len(input)):
        y_pred = max(0,w*input[i]  +   bias)
        total_loss = total_loss + (y_true[i] - y_pred) ** 2
    return total_loss/len(input)

weights = np.arange (-5,5,0.1)
losses = [loss(w) for w in weights]

plt.plot(weights, losses)
plt.show()