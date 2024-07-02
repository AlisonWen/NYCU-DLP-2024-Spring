import numpy as np
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
writer = SummaryWriter()
plt.rcParams["font.family"] = "Times"

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)
        if 0.1 * i == 0.5:
            continue
        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, y_pred, fig_name):
    plt.subplot(1, 2, 1)
    plt.title('Ground Truth', fontsize = 12)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1, 2, 2)
    plt.title('Predicted Result', fontsize=12)
    for i in range(x.shape[0]):
        if y_pred[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    if fig_name is not None:
        plt.savefig(fig_name + '.png')
    plt.show(block=True)
    
    

class Layer:
    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, input_grad):
        raise NotImplementedError

class FullyConnectedLayer(Layer):
    def __init__(self, input_shape, output_shape):
        self.W = np.random.randn(input_shape, output_shape) * 0.5
        self.b = np.zeros((1, output_shape))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
    
    def forward(self, input):
        self.input = input
        return np.dot(input, self.W) + self.b
    
    def backward(self, output_grad):
        self.dW = np.dot(self.input.T, output_grad)
        self.db = np.sum(output_grad, axis=0, keepdims=True)
        return np.dot(output_grad, self.W.T)
    
    def update(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db


class Network:
    def __init__(self, input_shape, hidden_size1, hidden_size2, output_shape):
        self.fc1 = FullyConnectedLayer(input_shape, hidden_size1)
        # self.act1 = Sigmoid()
        self.fc2 = FullyConnectedLayer(hidden_size1, hidden_size2)
        # self.act2 = Sigmoid()
        self.fc3 = FullyConnectedLayer(hidden_size2, output_shape)
        # self.act3 = Sigmoid()
        
    
    def forward(self, x):
        self.input = x
        x = self.fc1.forward(x)
        # x = self.act1.forward(x)
        x = self.fc2.forward(x)
        # x = self.act2.forward(x)
        x = self.fc3.forward(x)
        # x = self.act3.forward(x)
        return x
    
    def backward(self, Y):
        grad = self.forward(self.input) - Y
        # grad = self.act3.backward(grad)
        grad = self.fc3.backward(grad)
        # grad = self.act2.backward(grad)
        grad = self.fc2.backward(grad)
        # grad = self.act1.backward(grad)
        grad = self.fc1.backward(grad)
        return grad
    
    def update(self, learning_rate):
        self.fc1.update(learning_rate)
        self.fc2.update(learning_rate)
        self.fc3.update(learning_rate)
    
    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)
    
    def train(self, X, Y, learning_rate, max_iter):
        for i in range(max_iter):
            self.forward(X)
            self.backward(Y)
            self.update(learning_rate)
            if i % 100 == 0:
                loss = np.mean(np.square(Y - self.forward(X)))
                accuracy = (np.abs(self.predict(X) == Y)).sum() / Y.shape[0]
                # print('*prediction:', self.act3.output.flatten())
                print(f"Iteration: {i}, Loss: {loss:.4f}, Acc:{accuracy:.4f}")
                # writer.add_scalar('XOR', accuracy, i)
    
    def evaluate(self, X, Y, fig_name=None):
        predictions = self.predict(X)
        # precision = precision_score(Y, predictions) * 100
        accuracy = ((np.abs(self.predict(X) == Y)).sum() / Y.shape[0]) * 100
        print(f'Precision: {accuracy:.4f}%')
        print(self.forward(X).flatten())
        print('pred:', predictions.flatten())
        print('truth', Y.flatten())
        show_result(X, Y, predictions, fig_name)
        
    def save_weights(self, filename):
        np.save(f'{filename}_weights_1.npy', self.fc1.W)
        np.save(f'{filename}_biases_1.npy', self.fc1.b)
        np.save(f'{filename}_weights_2.npy', self.fc2.W)
        np.save(f'{filename}_biases_2.npy', self.fc2.b)
        np.save(f'{filename}_weights_3.npy', self.fc3.W)
        np.save(f'{filename}_biases_3.npy', self.fc3.b)
        
    def load_weights(self, filename):
        self.fc1.W = np.load(f'{filename}_weights_1.npy')
        self.fc1.b = np.load(f'{filename}_biases_1.npy')
        self.fc2.W = np.load(f'{filename}_weights_2.npy')
        self.fc2.b = np.load(f'{filename}_biases_2.npy')
        self.fc3.W = np.load(f'{filename}_weights_3.npy')
        self.fc3.b = np.load(f'{filename}_biases_3.npy')
        

# The generate_linear function remains the same.

if __name__ == "__main__":
    np.random.seed(42)
    X, Y = generate_linear(n=100)
    model = Network(input_shape=2, hidden_size1=16, hidden_size2=4, output_shape=1)
    # model = Network(input_shape=2, hidden_size1=128, hidden_size2=64, output_shape=1)
    model.train(X, Y, learning_rate=0.01, max_iter=5000)
    model.evaluate(X, Y, 'Linear Classification')
    # model.save_weights('LinearClassification')
    
    X, Y = generate_XOR_easy()
    print(np.shape(X), np.shape(Y))
    model = Network(input_shape=2, hidden_size1=128, hidden_size2=64, output_shape=1)
    # model = Network(input_shape=2, hidden_size1=2, hidden_size2=2, output_shape=1)
    model.train(X, Y, learning_rate=0.001, max_iter=50000)
    model.evaluate(X, Y, 'XOR')
    # model.save_weights('XOR')
