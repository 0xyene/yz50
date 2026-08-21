#3. Basit bir loss fonksiyonu yaz. 
#MSE 

y_true = [1,2,3]
y_pred = [4,5,6]

def mse(y_true,y_pred):
    n=len(y_true)
    squared_errors = []
    for i in range(len(y_true)):
        squared_error= (y_true[i] - y_pred[i]) ** 2 
        squared_errors.append(squared_error)
    return((sum(squared_errors))/n)
print(mse(y_true,y_pred))