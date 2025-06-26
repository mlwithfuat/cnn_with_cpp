# 🧠 Simple C++ Convolutional Neural Network (CNN)

This repository contains a **from-scratch implementation** of a Convolutional Neural Network (CNN) written entirely in C++. It is designed as a minimal, **educational example** for understanding core CNN operations **without relying on external libraries** or large datasets like MNIST.

> 🚀 Ideal for learning purposes, fast prototyping, and exploring how CNNs work under the hood in C++.

---

## 📁 Project Structure

- **`matrix.h`**  
  Implements a basic `Matrix` class with essential linear algebra operations:
  - Matrix addition, subtraction, multiplication
  - Element-wise operations
  - Transpose
  - **CNN-specific functions**: 2D convolution, max pooling, flattening, reshaping
  - Activation functions: `ReLU`, `Sigmoid`, `Tanh`, `Softmax`

- **`cnn_layers.h`**  
  Defines CNN layer classes:
  - `Layer` (abstract base class)
  - `ConvolutionalLayer`
  - `ActivationLayer`
  - `MaxPoolingLayer`
  - `FlattenLayer`
  - `FullyConnectedLayer`
  - `DropoutLayer`  
  Each includes `forward()` and `backward()` methods.

- **`cnn.h`**  
  Core `CNN` class:
  - Manages layers
  - Executes forward & backward propagation
  - Computes losses (`categorical cross-entropy`, `mean squared error`)
  - Evaluates accuracy
  - Contains training loop logic

- **`main.cpp`**  
  The application entry point:
  - Generates dummy training and test data
  - Constructs a simple CNN model
  - Trains and evaluates the model
  - Prints performance metrics

---

## ⚙️ Prerequisites

- C++ compiler with **C++17** support (e.g., GCC 7+, Clang 5+)
- Standard C++ Library (STL)

---

## 🛠️ Getting Started

### 1. Setup

Clone the repo or copy the following files into a single directory:

- `matrix.h`
- `cnn_layers.h`
- `cnn.h`
- `main.cpp`

Ensure your version of `cnn.h` includes:
```cpp
#include <numeric>
#include <random>
