from torch import nn
from torch.optim import Adam, SGD, Adagrad


class DQN(nn.Module):
    def __init__(self, activation, layers, weights='xunif', optim='Adam', learning_rate=1e-3):
        super(DQN, self).__init__()
        self.layers = layers
        assert len(self.layers) >= 2, "There needs to be at least an input and output "
        self.learning_rate = learning_rate

        # Make activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sig':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

        # Make weights function
        if weights == 'xunif':
            self.weights_init = nn.init.xavier_uniform_
        elif weights == 'xnorm':
            self.weights_init = nn.init.xavier_normal_
        elif weights == 'unif':
            self.weights_init = nn.init.uniform_
        elif weights == 'norm':
            self.weights_init = nn.init.normal_
        else:
            print('No known weight initialisation provided. Using Xavier Uniform')
            self.weights_init = nn.init.xavier_uniform_

        # Make Layers
        self.nn_model = self.apply_layers()

        # Make optimisation function
        if optim == 'Adam':
            self.optim = Adam
        elif optim == 'SGD':
            self.optim = SGD
        elif optim == 'Adagrad':
            self.optim = Adagrad
        else:
            print('No known optimiser provided. Using Adam.')
            self.optim = Adam

        self.optim = self.optim(self.parameters(), lr=self.learning_rate)

    def make_weights_bias(self, layer):
        # Initialise weights randomly and set biases to zero
        self.weights_init(layer.weight)
        nn.init.zeros_(layer.bias)

    def apply_layers(self):
        layer_list = []

        for k in range(len(self.layers) - 1):
            layer = nn.Linear(in_features=self.layers[k], out_features=self.layers[k + 1])
            self.make_weights_bias(layer)
            layer_list.append(layer)
            if k < len(self.layers) - 1:
                # No activation function applied to the output layer
                layer_list.append(self.activation)

        return nn.Sequential(*layer_list)

    def forward(self, state):

        return self.nn_model(state)


if __name__ == '__main__':
    pass
