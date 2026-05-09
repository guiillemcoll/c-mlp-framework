# c-mlp-framework

A lightweight, fully functional Multi-Layer Perceptron (MLP) neural network built entirely from scratch in pure C. 

This project was developed to gain a deep, low-level understanding of the mathematics and memory management behind artificial intelligence, avoiding the abstraction of modern frameworks like PyTorch or TensorFlow.

## Key Features

* **Built from Scratch:** No external machine learning libraries used. Pure algorithmic logic and C standard libraries.
* **Custom Architecture:** Dynamic initialization of hidden layers, neuron counts, and learning rates.
* **Core Mathematics:** Manual implementation of Forward Propagation and Backpropagation algorithms.
* **Activation Functions:** Includes custom derivatives for `ReLU` (hidden layers) and `Sigmoid` (output layer).
* **Persistent Models:** Capability to train a model, serialize the weights/biases to a `.txt` file, and load them later for inference.
* **Interactive CLI:** Built-in command-line interface to train, save, load, and test models seamlessly.

## Under the Hood

The network is built using standard C structs, optimizing memory management and ensuring low-latency execution. It handles data normalization automatically and updates weights using stochastic gradient descent (SGD) based principles.

## Compilation and Usage

To compile the code, you need to link the math library (`-lm`) because of the use of exponential and square root functions during weight initialization and backpropagation:
```bash
gcc -o mlp-framework.c -lm
./mlp-framework
