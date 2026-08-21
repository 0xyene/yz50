#2. Birden fazla nörondan oluşan küçük bir katman kur, forward pass'i buna genişlet. 
#Aktivasyon fonksiyonu relu, 3 neuron, tek layer, 3 input neuron bir neural net icin forward pass


inputs =  [1 ,2 ,3]
weights = [[0.6, -0.3, 1.1], 
           [0.5, -0.2, 0.9],
           [0.4, -0.1, -0.7]]
biases =  [0.3,0.42,-0.58]

def frwd_pass(inputs,weights,biases):
    outputs=[]
    for i in range(len(weights)):
        total = 0

        for j in range(len(inputs)):
            total = inputs[j]*weights[i][j] + total
        total = biases[i] + total
        outputs.append(max(0, total))
    return(outputs)
print(frwd_pass(inputs,weights,biases))