import numpy as np
import time
from Neuron import Neuron
from ImageProc import ImageProc
import os
from activation_functions import Heaviside_with_param


class Perceptron:
    def __init__(self,n_a):
        print("Perceptron: creating...")

        self.n_s = 28 * 28 # S-layer, image pixel count
        self.n_a = n_a # A-layer
        self.n_r = 10 # R-layer 

        self.b_R = 0.5
        self.b_A = 0.5

        self.R_layer = self.gen_result_R_neurons(self.n_r)
        self.A_layer = self.gen_random_A_weights()

        self.f_S_A = np.vectorize(Heaviside_with_param)

        print("Perceptron: creation OK")
    

    def get_weights(self):
        return np.zeros((self.n_a, 1))


    def gen_random_A_weights(self):
        element_set = (-1, 0, 1)
        return np.random.choice(element_set,(self.n_a, self.n_s))


    def gen_result_R_neurons(self, n):
        a = []
        
        for _ in range(n):
            a.append(Neuron(self.get_weights(), self.b_R))

        return a


    def from_S_to_A(self, example):
        vsum = self.A_layer.dot(example.T)
        return self.f_S_A(vsum, self.b_A)


    def from_A_to_R(self, example):
        output_vector = []
        for neuron in self.R_layer:
            answ = neuron.vectorized_forward_pass(example)
            output_vector.append(answ)
        return output_vector


    def forward_pass(self, example):
        sa_example = self.from_S_to_A(example)
        return self.from_A_to_R(sa_example)


    def train_neuron(self, input_matrix, y, neuron, max_steps=1e4):
        proc_matrix = []
        for i in range(len(X)):
            proc_matrix.append(self.from_S_to_A(input_matrix[i]))
        errors = neuron.train_until_convergence(proc_matrix, y, max_steps)
        return errors


    def train_network(self, X, y_type):
        print("Perceptron: start training...")

        for i in range(len(self.R_layer)):
            y = np.array([np.array([1]) if i == el else np.array([0]) for el in y_type ])
            errors = self.train_neuron(X, y, self.R_layer[i])
            #print(errors)

        print("Perceptron: training OK")


if __name__ == "__main__":
    X = []
    y = []
    N = 20

    for i in range(10):
        I = ImageProc(N, "training/%d/" % i)
        X += I.get_pictures()
        y += [i] * N


    P = Perceptron(2000)
    start_training = time.time()
    P.train_network(X, y)
    print("Training complite in %fsec" % round(time.time() - start_training, 4))

    print("\nTesting results:")
    N = 25
    avg = 0
    for i in range(10):
        I = ImageProc(N, "testing/%d/" % i)
        X = I.get_pictures()
        success = 0
        for example in X:
            answ = P.forward_pass(example)
            if answ[i] == 1:
                success += 1
        avg += success 
        print("Digit %d," % i, "accuracy:", round(success / N * 100, 2), "%")
        
    print("Average accuracy:", round(avg / (10 * N)  * 100, 2), "%")
