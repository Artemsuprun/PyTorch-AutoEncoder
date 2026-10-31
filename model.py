#
# File:   model.py
# Author: Artem Suprun
# Date:   10/29/2026
# Description: A generic auto_encoder model implemented in PyTorch. 
#

import torch.nn as nn

class AutoEncoder(nn.Module):

    def __init__(self, encoder_dim, activation="relu", output_activation="None"):
        super().__init__()

        # Check dimensions are valid
        if len(encoder_dim) < 2:
            raise ValueError("Encoder dimensions must be at least 2 (input and latent)")
        
        # Activation functions
        activations = {
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
            "None": None
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}")
        self.act = activations[activation]
        self.act_name = activation
        if output_activation not in activations:
            raise ValueError(f"Unsupported output activation function: {output_activation}")
        self.output_act = activations[output_activation]

        # Build encoder layers (start off by flattening the inputs)
        encoder_layers = [nn.Flatten()]
        # loop through encoder dimensions and create linear layers with activations
        for i in range(len(encoder_dim) - 1):
            encoder_layers.append(nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < len(encoder_dim) - 2 and self.act is not None:  # No activation after last layer
                encoder_layers.append(self.act())

        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers (reverse of encoder)
        decoder_layers = []
        decoder_dim = list(reversed(encoder_dim))
        for i in range(len(decoder_dim) - 1):
            decoder_layers.append(nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if i < len(decoder_dim) - 2:  # No activation after last layer
                decoder_layers.append(self.act())
        # Add output activation after the last layer
        if self.output_act is not None:
            decoder_layers.append(self.output_act())
        
        self.decoder = nn.Sequential(*decoder_layers)

        # initalize random weights
        self._init_weights()

        # dimensions useful for debuging
        self.latent_dim = encoder_dim[-1]
        self.input_dim = encoder_dim[0]

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    # the action of passing the inputs through the model, while also 
    # flattening and reconstructing the original input images.
    def forward(self, x):
        original_shape = x.size()
        z = self.decode(self.encode(x))
        return z.view(original_shape)
    
    # initializing the weights to be random, including the biases.
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.act_name in ["relu", "leaky_relu"]:
                    nn.init.kaiming_uniform_(
                        module.weight,
                        nonlinearity="relu"
                    )
                else:
                    nn.init.xavier_uniform_(module.weight)
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def bottle_neck(self, x):
        return self.encoder(x)

