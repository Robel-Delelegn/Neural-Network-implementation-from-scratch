import numpy as np

num_data_points = 100
X = np.random.randint(32, 120, size=(num_data_points, 7))

# Generate random one-hot encoded labels for Y
Y = np.zeros((num_data_points, 4))
for i in range(num_data_points):
    Y[i, np.random.randint(0, 4)] = 1
class layer:

    def __init__(self,  neurons, layer_before=None, layer_next=None):
        self.layer_before = layer_before
        self.layer_next = layer_next
        self.neurons = neurons
        self.step = 0.1
        self.output = None
        self.error = None
        self.db = None
        self.dw = None

    def initialize(self):
        self.weights = np.random.randn(self.neurons, self.layer_before.neurons)
        self.biases = np.random.randn(1, self.neurons)

    def sigmoid(self, d):
        return 1/(1+np.exp(-d))

    def sig_der(self, q):
        return self.sigmoid(q)*(1-self.sigmoid(q))
    def relu(self, d):
        return np.maximum(0,d)

    def relu_der(self, q):
        return np.where(q > 0, 1, 0)
    def forward(self):
        z = self.layer_before.output.dot(self.weights.T) + self.biases
        if self.layer_next is None:
            self.output = self.sigmoid(z)
        else:
            self.output = self.relu(z)

    def backpropagation(self, x, y):
        if self.layer_next is None:
            self.error = (self.output-y)*self.sig_der(self.output)

        else:
            self.error = (self.layer_next.error.dot(self.layer_next.weights))*self.relu_der(self.output)

        self.db = np.sum(self.error, axis=0, keepdims=True)
        self.dw = self.error.T.dot(self.layer_before.output)
        self.weights -= self.step*self.dw/x.shape[0]
        self.biases -= self.step*self.db/x.shape[0]





layer0 = layer(neurons=7)
layer0.output = X
layer1 = layer(100, layer0)
layer1.initialize()
layer2 = layer(23, layer1)
layer2.initialize()
layer3 = layer(50, layer2)
layer3.initialize()
layer4 = layer(50, layer3)
layer4.initialize()
layer5 = layer(4, layer4)
layer5.initialize()

layer1.layer_next = layer2
layer2.layer_next = layer3
layer3.layer_next = layer4
layer4.layer_next = layer5
for i in range(5000):
    layer1.forward()
    layer2.forward()
    layer3.forward()
    layer4.forward()
    layer5.forward()
    a = np.argmax(layer5.output, axis=1)
    b = np.argmax(Y, axis=1)
    got_right = 0
    for i in range(len(a)):
        if [i] == b[i]:
            got_right += 1
    print(got_right/len(b))
    layer5.backpropagation(X, Y)
    layer4.backpropagation(X, Y)
    layer3.backpropagation(X, Y)
    layer2.backpropagation(X, Y)
    layer1.backpropagation(X, Y)