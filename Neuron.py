class Neuron:
    def __init__(self, w, b, lr = 1):
        self.lr = lr
        self.w = w
        self.b = b

    def vectorized_forward_pass(self, input_matrix):
        return (input_matrix.dot(self.w) + self.b > 0)
    
    def train_on_single_example(self, example, y):
        predict  = self.vectorized_forward_pass(example.T)[0][0]
        error = y[0] - predict

        error *= self.lr

        self.w += error * example
        self.b += error

        return abs(error) 


    def train_until_convergence(self, input_matrix, y, max_steps=1e4):
        i = 0
        errors = 1
        while errors and i < max_steps:
            i += 1
            errors = 0
            for example, answer in zip(input_matrix, y):
                example = example.reshape((example.size, 1))
                error = self.train_on_single_example(example, answer)
                errors += int(error) 
        return errors