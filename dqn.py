from typing import List, Optional
import torch
from torch import nn
from torch.optim import Adam, SGD, Adagrad


class DQN(nn.Module):
    def __init__(
        self,
        activation: str,
        layers: List,
        weights: Optional[str] = "xunif",
        optim: Optional[str] = "Adam",
        learning_rate: Optional[float] = 1e-3,
    ):
        """Deep Q Network Class

        Args:
            activation: Activation function to use in the network.
            layers: List of layer sizes to use in the network.
            weights: Weight initialisation method to use in the network defaults to Xavier Uniform.
            optim: Optimisation method to use in the network defaults to Adam.
            learning_rate: Learning rate to use in the network defaults to 1e-3.
        """

        super(DQN, self).__init__()
        self.layers = layers
        assert len(self.layers) >= 2, "There needs to be at least an input and output "
        self.learning_rate = learning_rate

        # Set Activation Function
        self.activation = self.set_activation(activation)

        # Make weights initialisation method
        self.weights_init = self.set_weights_init(weights)

        # Apply Layers
        self.nn_model = self.apply_layers()

        # Set optimisation function
        self.optim = self.set_optimizer(optim)

    @staticmethod
    def set_activation(activation: str) -> nn.Module:
        """Set activation function for the network

        Args:
            activation: Activation function to use in the network.

        Returns:
            nn.Module: Activation function module.
        """

        # Make activation function
        if activation == "relu":
            return nn.ReLU()
        elif activation == "sig":
            return nn.Sigmoid()
        elif activation == "tanh":
            return nn.Tanh()
        else:
            print("No known activation function provided. Using ReLU.")
            return nn.ReLU()

    @staticmethod
    def set_weights_init(weights: str) -> torch.nn.init:
        """Set weight initialisation method for the network

        Args:
            weights: Weight initialisation method to use in the network.

        Returns:
            torch.nn.init: Weight initialisation function.
        """
        if weights == "xunif":
            return nn.init.xavier_uniform_
        elif weights == "xnorm":
            return nn.init.xavier_normal_
        elif weights == "unif":
            return nn.init.uniform_
        elif weights == "norm":
            return nn.init.normal_
        else:
            print("No known weight initialisation provided. Using Xavier Uniform")
            return nn.init.xavier_uniform_

    def set_optimizer(self, optim: str) -> torch.optim:
        """Set optimisation method for the network

        Args:
            optim: Optimisation method to use in the network.

        Returns:
            torch.optim: Optimisation function.
        """

        if optim == "Adam":
            optim_fn = Adam
        elif optim == "SGD":
            optim_fn = SGD
        elif optim == "Adagrad":
            optim_fn = Adagrad
        else:
            print("No known optimiser provided. Using Adam.")
            optim_fn = Adam

        return optim_fn(self.parameters(), lr=self.learning_rate)

    def make_weights_bias(self, layer):
        """Initialise weights and biases for a layer

        Args:
            layer: Layer to initialise weights and biases for.
        """
        self.weights_init(layer.weight)
        nn.init.zeros_(layer.bias)

    def apply_layers(self):
        """Apply layers to the network

        Returns:
            nn.Sequential: Sequential model of the network.
        """
        layer_list = []
        for k in range(len(self.layers) - 1):
            layer = nn.Linear(
                in_features=self.layers[k], out_features=self.layers[k + 1]
            )
            self.make_weights_bias(layer)
            layer_list.append(layer)
            if k < len(self.layers) - 1:
                # No activation function applied to the output layer
                layer_list.append(self.activation)

        return nn.Sequential(*layer_list)

    def forward(self, state):
        """Forward pass through the network"""
        return self.nn_model(state)
