import numpy as np
from keras.datasets import fashion_mnist

class MultiLayerPerceptron():
    parameters = {}
    gradients = {}

    def __init__(self, train_data, train_labels, layer_sizes):
        self.train_data = train_data
        self.train_labels = train_labels
        self.no_of_samples = len(train_data)
        self.no_of_features = train_data.shape[1] #size of column of train_data
        self.no_of_classes = 10 #size of column of train_labels
        self.layer_sizes = layer_sizes

        MultiLayerPerceptron.parameters = {}
        norm = 1e3
        for i in range(len(layer_sizes) - 1):
            #input layer(n) and 1st hidden layer(m) will have weight vector w1->n*m and bias b1->1*m
            #init W to random values
            MultiLayerPerceptron.parameters[f'w_{i+1}'] = np.random.rand(layer_sizes[i], layer_sizes[i+1]) / norm
            #init B to 0
            MultiLayerPerceptron.parameters[f'b_{i+1}'] = np.zeros((layer_sizes[i+1]))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_d(self, x):
        return (x > 0) * 1
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def feed_forwards(self, input_data):
        self.layer_outputs = []
        for i in range(len(self.layer_sizes) - 1):
            if(i==0) :#input is our images
                #input_data->1*m,w->m*n,b->1*n
                pre_activation=np.matmul(input_data , MultiLayerPerceptron.parameters[f'w_{i+1}']) + MultiLayerPerceptron.parameters[f'b_{i+1}']
            else:
                #input data=outpu of last processed layer(layer_outputs[-1])
                pre_activation = np.matmul(self.layer_outputs[-1], MultiLayerPerceptron.parameters[f'w_{i+1}']) + MultiLayerPerceptron.parameters[f'b_{i+1}']

            if i < len(self.layer_sizes) - 2 :
                activation = self.relu(pre_activation) 
            else :
                activation=self.softmax(pre_activation)

            self.layer_outputs.append(activation)
        #print(self.layer_outputs[-1][-1][-1])

    def back_prop(self, input_data, true_label):
        dL_dA = self.layer_outputs[-1] - true_label
        for i in range(len(self.layer_sizes) - 2, -1, -1):
            dA_dZ = self.relu_d(self.layer_outputs[i]) if i < len(self.layer_sizes) - 2 else 1
            dZ_dW = input_data if i == 0 else self.layer_outputs[i-1]
            dL_dW = np.dot(dZ_dW.T, dL_dA * dA_dZ)
            dL_dB = np.sum(dL_dA * dA_dZ, axis=0)
            MultiLayerPerceptron.gradients[f'w_{i+1}'] = dL_dW
            MultiLayerPerceptron.gradients[f'b_{i+1}'] = dL_dB
            if i > 0:
                dL_dA = np.dot(dL_dA * dA_dZ, MultiLayerPerceptron.parameters[f'w_{i+1}'].T)

    def batch_gradient_descent(self, epochs=1000, batch_size=32, learning_rate=0.01):
        batch_size=self.no_of_samples
        no_of_batches = self.no_of_samples // batch_size

        for i in range(epochs):
            for j in range(no_of_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                Y_train = self.train_labels[start:end]
                X_train = self.train_data[start:end]

                self.feed_forwards(X_train)
                self.back_prop(X_train, Y_train)

                for k in range(len(self.layer_sizes) - 1):
                    MultiLayerPerceptron.parameters[f'w_{k+1}'] -= learning_rate * (MultiLayerPerceptron.gradients[f'w_{k+1}'] / batch_size)
                    MultiLayerPerceptron.parameters[f'b_{k+1}'] -= learning_rate * (MultiLayerPerceptron.gradients[f'b_{k+1}'] / batch_size)                


            if i == 0 or (i + 1) % 10 == 0:
                loss = self.cross_entropy_loss(self.layer_outputs[-1], Y_train)
                print(f'EPOCH NO: {i + 1} loss:{loss}')

    def train(self, epochs=50, batch_size=32, learning_rate=0.01):
        #this is mini batch gradient descent 
        #use this to train as this converges faster and gives better accuracy
        no_of_batches = self.no_of_samples // batch_size

        for i in range(epochs):
            for j in range(no_of_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                Y_train = self.train_labels[start:end]
                X_train = self.train_data[start:end]

                self.feed_forwards(X_train)
                self.back_prop(X_train, Y_train)

                for k in range(len(self.layer_sizes) - 1):
                    MultiLayerPerceptron.parameters[f'w_{k+1}'] -= learning_rate * (MultiLayerPerceptron.gradients[f'w_{k+1}'] / batch_size)
                    MultiLayerPerceptron.parameters[f'b_{k+1}'] -= learning_rate * (MultiLayerPerceptron.gradients[f'b_{k+1}'] / batch_size)

            if i == 0 or (i + 1) % 10 == 0:
                print(f'EPOCH NO: {i + 1}')

    def test(self, x, y):
        self.feed_forwards(x)
        loss = self.cross_entropy_loss(self.layer_outputs[-1], y)
        acc = self.accuracy(self.layer_outputs[-1], y)
        print(f'TEST LOSS: {loss:.4f} ACCURACY: {acc*100:.4f}')

    def cross_entropy_loss(self, pred, actual):
        return -np.sum(actual * np.log(pred + 1e-9)) / len(pred)
    
    def accuracy(self, pred, truth):
        return (np.argmax(truth, axis=1) == np.argmax(pred, axis=1)).mean()

#Load dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
total_class=max(train_labels)

train_images = train_images.reshape(train_images.shape[0], -1) / 255.0
test_images = test_images.reshape(test_images.shape[0], -1) / 255.0

#change trainlabel=2 to oneHot=[0 0 1 0 0 0 0 0 0 0]
train_labels_one_hot = np.eye(10)[train_labels]
test_labels_one_hot = np.eye(10)[test_labels]

# print(train_labels.ndim)
# y_train_one_hot = np.eye(10)[train_labels]
# print(train_labels[0])
# print(y_train_one_hot[0])

#We can give any number of layers and their sizes here
layer_sizes = [train_images.shape[1],128, 128,128,10]  # Input, hidden1, hidden2,hidden3 output

#init , train and test our model
mlp = MultiLayerPerceptron(train_images, train_labels_one_hot, layer_sizes)
mlp.train()
mlp.test(test_images, test_labels_one_hot)
