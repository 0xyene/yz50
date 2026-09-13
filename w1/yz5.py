#5. Sayısal türev (numerical derivative) ile basit bir gradient descent döngüsü kur: parametreyi küçük adımlarla güncelleyerek loss'u düşür. 

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

w = 1 
learning_rate = 0.01
h = 0.00001
losses = []

for i in range(1000):
    losses.append(loss(w))
    gradient = (loss(w+h)-loss(w))/h
    w = w - learning_rate * gradient

print("best w =", w )
print("best loss=", loss(w))

plt.plot(losses)
plt.show()
