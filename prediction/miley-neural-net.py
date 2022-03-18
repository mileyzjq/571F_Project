import random
import math
import numpy as np
from collections import defaultdict

class NeuralNet():
    def __init__(self):
        self.input_num = 7
        self.hidden_num = 12
        self.output_num = 1
        self.learning_rate = 0.2
        self.momentum = 0.9
        self.input_layer = []
        self.hidden_layer = []
        self.output_layer = []
        self.w1 = []
        self.w2 = []
        self.delta1 = []
        self.delta2 = []
        self.train_x1 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.train_y1 = [0, 1, 1, 0]
        self.train_x2 = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        self.train_y2 = [-1, 1, 1, -1]
        for i in range(self.input_num + 1):
            self.input_layer.append(0)
        for i in range(self.hidden_num + 1):
            self.hidden_layer.append(0)
        for i in range(self.output_num):
            self.output_layer.append(0)
        self.initialize_weights()

    def sigmoid(self, x):
        return 2 / (1 + math.exp(-x)) - 1

    def initialize_weights(self):
        for i in range(self.input_num + 1):
            tmp = []
            for j in range(self.hidden_num):
                tmp.append(random.uniform(-0.5, 0.5))
            self.w1.append(tmp)

        for i in range(self.hidden_num + 1):
            tmp = []
            for j in range(self.output_num):
                tmp.append(random.uniform(-0.5, 0.5))
            self.w2.append(tmp)

        for i in range(self.input_num + 1):
            tmp = []
            for j in range(self.hidden_num):
                tmp.append(0)
            self.delta1.append(tmp)

        for i in range(self.hidden_num + 1):
            tmp = []
            for j in range(self.output_num):
                tmp.append(0)
            self.delta2.append(tmp)

    def initialize_input_layer(self, feature):
        i = 0
        for v in feature:
            self.input_layer[i] = float(feature[v])
            #print("feature: " + str(self.input_layer[i]))
            i += 1
        # for i in range(self.input_num):
        #     self.input_layer[i] = feature[i]
        self.input_layer[self.input_num] = 1
        self.hidden_layer[self.hidden_num] = 1

    def forward_propagation(self, feature):
        self.initialize_input_layer(feature)
        for j in range(self.hidden_num):
            for i in range(self.input_num + 1):
                self.hidden_layer[j] += self.w1[i][j] * self.input_layer[i]
            self.hidden_layer[j] = self.sigmoid(self.hidden_layer[j])

        for j in range(self.hidden_num + 1):
            self.output_layer[0] += self.w2[j][0] * self.hidden_layer[j]
        self.output_layer[0] = self.sigmoid(self.output_layer[0])

    def backward_propagation(self, error):
        delta_output = []
        delta_hidden = []

        tmp = 0.5 * (1 - math.pow(self.output_layer[0], 2)) * error
        delta_output.append(tmp)

        for k in range(self.output_num):
            for j in range(self.hidden_num + 1):
                self.delta2[j][k] = self.momentum * self.delta2[j][k] + self.learning_rate * delta_output[k] * self.hidden_layer[j]
                self.w2[j][k] += self.delta2[j][k]

        for j in range(self.hidden_num):
            tmp = 0
            for k in range(self.output_num):
                tmp += self.w2[j][k] * delta_output[k]
            tmp = tmp * 0.5 * (1 - math.pow(self.hidden_layer[j], 2))
            delta_hidden.append(tmp)

        for j in range(self.hidden_num):
            for i in range(self.input_num + 1):
                self.delta1[i][j] = self.momentum * self.delta1[i][j] + self.learning_rate * delta_hidden[j] * self.input_layer[i]
                self.w1[i][j] += self.delta1[i][j]

    def evaluate(self, feature, expected_value):
        self.forward_propagation(feature)
        error = float(expected_value) - self.output_layer[0]
        loss = math.pow(error, 2) / 2
        self.backward_propagation(error)
        return (self.output_layer[0], loss)

    def test(self, feature, expected_value):
        self.forward_propagation(feature)
        error = float(expected_value) - self.output_layer[0]
        loss = math.pow(error, 2) / 2
        return (self.output_layer[0], loss)

    def train_neural_net(self):
        epoch = 0
        total_error = 100
        self.initialize_weights()

        while total_error >= 0.05 and epoch < 10000:
            total_error = 0
            for i in range(4):
                score, loss = self.evaluate(self.train_x1[i], self.train_y1[i])
                total_error += loss
            total_error /= 2
            print("epoch: " + str(epoch) + " error: " + str(total_error))
            epoch += 1

# nn = NeuralNet()
# nn.train_neural_net()
