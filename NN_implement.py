import numpy as np 
from random import random

class ANN(object):
    
    def __init__(self, num_inputs=2, hidden=[3, 3], num_outputs=1):
        self.num_inputs = num_inputs
        self.hidden = hidden
        self.num_outputs = num_outputs
        
        layers = [num_inputs] + hidden + [num_outputs]
        
        weights = []
        for w in range(len(layers)-1):
            w_matrix = np.random.rand(layers[w], layers[w+1])
            weights.append(w_matrix)
        self.weights = weights
            
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations
        
        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros(layers[i])
            derivatives.append(d)
        self.derivatives = derivatives
            
    def forward_propagate(self, inputs):
        # multiply input and weights
        # add them up
        # activate it
        
        activation = inputs
        self.activations[0] = activation
        for i, w in enumerate(self.weights):
            weighted_inputs = np.dot(activation, w)
            activation = self._sigmoid(weighted_inputs)
            """ print("Activation {} shape: {}".format(i, np.shape(activation))) """
            self.activations[i+1] = activation
        return self.activations[-1]
        
    def backpropagate(self, error):
        
        for i in reversed(range(len(self.derivatives))):
            activation = self.activations[i+1]
            # error*self._sigmoid_derivative(activation)*self.activations[i] is the general formula for fetching derivatives
            # reshape these 1D arrays to 2D arrays as derivatives are 2D because diff wrt to weight matrices which are 2D 
            d_error = error*self._sigmoid_derivative(activation)
            re_d_error = d_error.reshape(d_error.shape[0], -1).T # -1 tells numpy to automatically use a suitable column dimension
            current_activations = self.activations[i]
            current_activations = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations, re_d_error) # activation dot delta, since we need the right dimension of derivative(identical dimension to weight matrix)
            error = np.dot(d_error, self.weights[i].T) # update the error for looping through the layers upto input layer. Eq to update each subsequent layer will have the error variable contain the previously calculated d_error dotted with the weight matrix(transposed because d_error will have dim of activation, but in case of weight matrix, dimension of the column will match dimension of activation, hence done so)
            # uncomment the below lines to figure out why transposes are required as it will aid in dimension matching for understanding purposes
            """ print("_______________________")
            print("error {} shape: {}".format(i, np.shape(error)))
            print("self.weights[i] {} shape: {}".format(i, np.shape(self.weights[i])))
            print("d_error {} shape: {}".format(i, np.shape(d_error)))
            print()
            print("re_d_error {} shape: {}".format(i, np.shape(re_d_error)))
            print("re_curr_act {} shape: {}".format(i, np.shape(current_activations)))
            print("self._sigmoid_derivative(activation) {} shape: {}".format(i, np.shape(self._sigmoid_derivative(activation))))
            print("self.activations[i+1] {} shape: {}".format(i, np.shape(self.activations[i+1])))
            print("current_activations {} shape: {}".format(i, np.shape(current_activations)))
            print("_______________________") """
            
    def train(self, inputs, outputs, epochs, learning_rate):
        for e in range(epochs):
            error_sum = 0
            for i, input_val in enumerate(inputs):
                target = outputs[i]
                pred = self.forward_propagate(input_val)
                """ print("Iteration", i, input_val, "Target:",target," pred:", pred) """
                error = target - pred
                self.backpropagate(error)
                
                self.gradient_descent(learning_rate)
                error_sum += self._mse(target, pred)
                error_sum = error_sum / len(outputs)
            print("Epoch {} error: {} ".format(e, error_sum))
            
    def _mse(self, target, pred):
        return np.average((target - pred)**2)
                
                
    def gradient_descent(self, learning_rate): # weight update rule, update existing weights with derivatives and learning rate
        for i in range(len(self.weights)):
            self.weights[i] += self.derivatives[i]*learning_rate
    
    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)
    
    def _sigmoid(self, x):
        return np.divide(1.0, np.add(1, np.exp(-x)))
        # return 1.0 / (1 + np.exp(-x))
        
if __name__ == "__main__":
    ann_model = ANN(2, [4], 1)
    inputs = np.array([[random()/2 for _ in range(2)] for _ in range(1000)]) # create 1000 elements, each element is an array of 2 numbers
    prediction_input = np.array([0.25, 0.45])
    prediction_target = np.array([0.5])
    targets = np.array([[i[0] + i[1]] for i in inputs]) # for each element in input, summation is the value for the target
    ann_model.train(inputs=inputs, outputs=targets, epochs=50, learning_rate=0.1)
    prediction = ann_model.forward_propagate(prediction_input) # prediction after model is trained
    print("predicted output for {} is {}".format(prediction_input, prediction))
    """ inputs = np.array([0.2, 0.3])
    output = np.array([0.5])
    pred = ann_model.forward_propagate(inputs)
    error = output - pred
    ann_model.backpropagate(error) """