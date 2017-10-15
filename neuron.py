import numpy as np
from scipy.special import expit


class Weight:

    def __init__(self, value=None, input_node=None, output_node=None):
        self.input = input_node
        self.output = output_node
        if value is None:
            self.value = np.random.random()
        else:
            self.value = value


class Neuron:

    def __init__(self, activation_func, input_count=0):
        self.activation_func = activation_func
        self.weights = []
        self.bias = np.random.random()
        self.value = None
        for i in range(input_count):
            self.weights.append(Weight(output_node=self))

    def evaluate(self):
        inputs = [w.input.evaluate() for w in self.weights]
        sum = 0.0
        for i, val in enumerate(inputs):
            sum += val * self.weights[i].value
        sum += self.bias
        self.value = self.activation_func(sum)
        return self.value

    def clear(self):
        """ Clear the memory of the neuron of its last value. """
        self.value = None


class InputNode(Neuron):

    def __init__(self):
        super().__init__(lambda x: x, 1)

    def set_value(self, value):
        self.value = value

    def evaluate(self):
        return self.value


class SingleHiddenFFNN:

    def __init__(self, inputs_count, hidden_units_count, outputs_count, activ_func=expit):
        self.input_nodes = [InputNode() for i in range(inputs_count)]
        self.hidden_nodes = [Neuron(activ_func) for i in range(hidden_units_count)]
        self.output_nodes = [Neuron(activ_func) for i in range(outputs_count)]

        # Tie the layers together with weights
        for inpt in self.input_nodes:
            for outpt in self.hidden_nodes:
                wgt = Weight(input_node=inpt, output_node=outpt)
                outpt.weights.append(wgt)

        for inpt in self.hidden_nodes:
            for outpt in self.output_nodes:
                wgt = Weight(input_node=inpt, output_node=outpt)
                outpt.weights.append(wgt)

        self.previous_weight_deltas = None

    def display_weights(self):
        print('----- Hidden Node Weights -----')
        for node in self.hidden_nodes:
            print([w.value for w in node.weights])

        print('----- Output Node Weights -----')
        for node in self.output_nodes:
            print([w.value for w in node.weights])

    def evaluate(self, inputs):
        if len(inputs) is not len(self.input_nodes):
            raise ValueError('Incorrect Number of inputs: received {}, should be {}'.format(len(inputs), len(self.input_nodes)))
        # Set the value of network inputs
        for i, inpt in enumerate(self.input_nodes):
            inpt.set_value(inputs[i])
        # get value of outputs
        result = []
        for outpt in self.output_nodes:
            result.append(outpt.evaluate())
        return result

    def get_hidden_error(self, hidden_node, output_deltas):
        hidden_error = 0
        for k, output in enumerate(self.output_nodes):
            for weight in output.weights:
                if hidden_node is weight.input:
                    hidden_error += weight.value * output_deltas[k]
        return hidden_error

    def backprop(self, errors, eta=0.1, alpha=0.0):
        """
        errors is a vector of the error of each output neuron.
        Assumes evaluate has already been run for this epoch.
        """

        if self.previous_weight_deltas is None:
            output_weights_dim = (len(self.hidden_nodes),len(self.output_nodes))
            hidden_weights_dim = (len(self.input_nodes),len(self.hidden_nodes))
            self.previous_weight_deltas = [np.zeros(hidden_weights_dim), np.zeros(output_weights_dim)]

        deltas = []
        # update weights in output layer
        for j, node in enumerate(self.output_nodes):
            delta = errors[j] * node.value * (1.0 - node.value)
            deltas.append(delta)
            for i, weight in enumerate(node.weights):
                new_w_delta = eta * delta * weight.input.value + alpha * self.previous_weight_deltas[1][i,j]
                weight.value += new_w_delta
                self.previous_weight_deltas[1][i,j] = new_w_delta
            node.bias += eta * delta

        # update weights in hidden layer
        for j, node in enumerate(self.hidden_nodes):
            delta = self.get_hidden_error(node, deltas) * node.value * (1.0 - node.value)
            for i, weight in enumerate(node.weights):
                new_w_delta = eta * delta * weight.input.value + alpha * self.previous_weight_deltas[0][i,j]
                weight.value += new_w_delta
                self.previous_weight_deltas[0][i,j] = new_w_delta
            node.bias += eta * delta

if __name__ == "__main__":
    bar = SingleHiddenFFNN(5, 7, 2)
    result = bar.evaluate((0, 1, 7, 42, 11))
    bar.display_weights()
    print(result)

