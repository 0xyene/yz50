#1. Python ile tek nöron forward pass yaz, kütüphane kullanmadan. 
#Aktivasyon fonksiyonu relu, 1 neuron, tek layer, 3 input neuron bir neural net icin forward pass
inputs = [1 ,2 ,3]
weights = [0.6, -0.3, 1.3]
bias = 0.3

def frwd_pass(inputs,weights,bias):

    total = 0

    for i in range(len(weights)):
        total = inputs[i]*weights[i] + total

    return max(0,total + bias)

print(frwd_pass(inputs,weights,bias))