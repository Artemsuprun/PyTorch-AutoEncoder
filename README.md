# PyTorch-AutoEncoder

A PyTorch implementation of a fully connected autoencoder designed to reconstruct simple images.

The model uses an encoder-decoder architecture to naturally reconstruct images.

## Overview

An autoencoder is an unsupervised learning model that learns to reproduce its input. It does this by compressing input data into smaller dimensions and then attempts to reconstruct it from the bottleneck. This project implements a configurable autoencoder using PyTorch, with support for different activation functions, configurable layer dimensions, and image reconstruction through a command-line interface.

## Architecture

The model consists of two primary parts:

- Encoder: Reduces the dimensionality of the input until it reaches the bottleneck.
- Decoder: Reverses the encoder's structure to reconstruct the original input from the bottleneck.

The bottleneck encourages the model to learn compressed features of the input, which helps it reconstruct it.

(insert an image of the model here Artem!, not ergent but should do it.)

*This model's layers are configurable in the encoder_dim parameter.

## Features

- Configurable encoder and decoder dimensions.
- Support ReLU, Sigmoid, Tanh, and Leaky ReLU activation functions.
- Configurable output activation.
- Access to the bottleneck with bottle_neck().
- Image reconstruction through the command-line interface.

## Project structure

- model.py: Where the autoencoder model lives.
- train.py: Used to train the model.
- predict.py:
- run.py:
- run_test.py:
- checkpoints/:
- data/:
- outputs/:

## Install

Clone the repo:

```bash
git clone https://github.com/Artemsuprun/PyTorch-AutoEncoder.git
cd PyTorch-AutoEncoder/
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Use

In order to begin using the model, you need to first train it. You can type this in for the default, which is to deblur the images from the MNIST dataset.

```bash
python train.py
```

Afterwards, the project includes a command-line interface for running inference on an image using the trained model.

```bash
python run.py --image path/to/image
```

This runs the input image through the autoencoder and displays the original and reconstructed image at the end. 

## Command-Line Arguments

| Arg | Required | Description |
| --- | -------- | ----------- |
| --imge | Yest | Path to the input image. |
| --checkpoint | No | Path to the trained model checkpoint. |
| --output | No | Path where the reconstructed image will be saved. |
| --no-display | No | Disable the image visualization window. |

## Example

Specify a custom checkpoint:

```bash
python run.py --image input.png --checkpoint checkpoints/autoencoder.pth
```

Save the reconstructed image:

```bash
python run.py --image input.png --output output/reconstruction.png
```

Run inference without displaying the images:

```bash
python run.py --image input.png --no-display
```

## Model Config

```python
from model import AutoEncoder

model = AutoEncoder(
    encoder_dim=[784, 128, 64, 32],
    activation="relu",
    output_activation="sigmoid"
)
```

This creates an encoder with the dimensions: 

784 -> 128 -> 64 -> 32 -> 32 -> 64 -> 128 -> 784

## Prerequisites 

- Python
- PyTorch
- Torchvision
- Matplotlib
- Pillow

## License

This project is licensed under the **Apache 2.0 license**.
