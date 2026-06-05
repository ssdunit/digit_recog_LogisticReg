import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

fetchdata = fetch_openml('mnist_784',version=1,as_frame=False,parser='auto')
X_labels = fetchdata.data
y = fetchdata.target.astype(int)

X_labels = X_labels/255.0

X_train,X_test,y_train,y_test = train_test_split(X_labels,y,test_size=0.2,random_state=42)

def one_hot_encode(labels):
    max_size = labels.max()+1
    encoded_arr = np.zeros((labels.size,max_size),dtype=float)
    encoded_arr[np.arange(labels.size),labels]=1.0
    return encoded_arr

Y_train_encoded = one_hot_encode(y_train)

num_features = X_train.shape[1]
num_classes = 10
hidden_class = 128

W1 = np.random.randn(num_features,hidden_class)*0.01
b1 = np.zeros((1,hidden_class),dtype=float)
W2 = np.random.randn(hidden_class,num_classes)*0.01
b2 = np.zeros((1,num_classes),dtype=float)

learning_rate = 1.05
epochs = 300

def softmax(Z):
    exp_z = np.exp(Z-np.max(Z,axis=1,keepdims=True))
    return exp_z/np.sum(exp_z,axis=1,keepdims=True)

def ReLU(Z):
    return np.maximum(0,Z)

def ReLU_Der(Z):
    return Z>0


m = X_train.shape[0] #56000

for epoch in range(epochs):
    Z1 = np.dot(X_train,W1) + b1
    A1 = ReLU(Z1)

    Z2 = np.dot(A1,W2)+b2
    A2 = softmax(Z2)

    dZ2 = A2-Y_train_encoded
    
    dW2 = (1/m)*np.dot(A1.T,dZ2)
    db2 = (1/m)*np.sum(dZ2,axis=0,keepdims=True)

    dZ1 = np.dot(dZ2,W2.T)*ReLU_Der(Z1)
    dW1 = (1/m)*np.dot(X_train.T,dZ1)
    db1 = (1/m)*np.sum(dZ1,axis=0,keepdims=True)

    W2 = W2-(learning_rate*dW2)    
    b2 = b2-(learning_rate*db2)
    W1 = W1-(learning_rate*dW1)
    b1 = b1-(learning_rate*db1)

    if (epoch+1)%5 == 0:
        loss = -np.mean(np.sum(Y_train_encoded * np.log(A2 + 1e-15), axis=1))
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss:.4f}")

Z1_test = np.dot(X_test,W1)+b1
A1_test = ReLU(Z1_test)
Z2_test = np.dot(A1_test,W2)+b2
A2_test = softmax(Z2_test)

predictions = np.argmax(A2_test,axis=1)

accuracy = np.mean(predictions == y_test)
print(f"Final Test Accuracy: {accuracy * 100:.2f}%")

np.savez("finaldataset.npz",W1=W1,b1=b1,W2=W2,b2=b2)

