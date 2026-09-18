# Small Neural Networks and LLMs from Scratch in C

A learning project for building neural networks—and eventually a small language model—from scratch in C.

The goal is not to build the fastest neural-network framework or the most capable model. The goal is to understand, implement, and verify the machinery underneath modern neural networks with as little hidden behind libraries as practical.

We will begin with individual neurons and basic gradient descent, build up to a handwritten-digit classifier using the MNIST dataset, and eventually reuse the same foundations to construct a small transformer-based language model.

## Goals

This project aims to:

* Implement the important neural-network operations ourselves.
* Understand the mathematics behind each operation.
* Connect the mathematical notation directly to C code.
* Understand forward propagation and backpropagation rather than relying on automatic differentiation.
* Build basic matrix and tensor operations from scratch.
* Learn how neural-network parameters are initialized, trained, and evaluated.
* Train a classifier on handwritten digits.
* Eventually build and train a small transformer language model.
* Keep dependencies to a minimum.

Performance optimization is secondary to clarity and correctness, especially during the early stages.

## Philosophy

The project follows a simple progression:

```text
derive the math
      |
      v
implement the simplest version
      |
      v
inspect intermediate results
      |
      v
verify numerically
      |
      v
understand the implementation
      |
      v
extract reusable components
```

We intentionally avoid introducing abstractions before they are useful.

For example, the first neuron may simply contain:

```c
float z = w0 * x0 + w1 * x1 + b;
```

Only after understanding and testing the underlying operation will we generalize it into vector, matrix, layer, or tensor operations.

## Dependencies

Initially, the project uses only:

* C17
* the C standard library
* the C math library (`libm`)
* Make
* GCC or Clang
* Git

Useful development tools include:

* GDB or LLDB
* AddressSanitizer
* UndefinedBehaviorSanitizer
* Valgrind, where available

The initial implementation deliberately avoids:

* PyTorch
* TensorFlow
* NumPy
* BLAS
* CUDA
* automatic differentiation libraries
* neural-network frameworks

These may be studied or introduced later for comparison, but the core implementations will be written by hand.

## Roadmap

The project will grow in small stages.

### Stage 1 — A Single Neuron

Implement a neuron with a small number of inputs:

```text
x0 ---- w0 --\
              \
               +---- weighted sum + bias ---- prediction
              /
x1 ---- w1 --/
```

For two inputs:

```text
z = w0*x0 + w1*x1 + b
```

From this simple example we will study:

* weights and biases
* forward propagation
* loss functions
* derivatives
* the chain rule
* gradient descent
* numerical gradient checking

### Stage 2 — Vectors and Matrices

Implement the basic numerical operations needed by neural networks:

* vectors
* matrices
* indexing
* dot products
* matrix-vector multiplication
* matrix multiplication
* memory allocation and ownership

A matrix will initially be represented by a simple structure similar to:

```c
typedef struct {
    size_t rows;
    size_t cols;
    float *data;
} Matrix;
```

No external linear-algebra library will perform these operations for us.

### Stage 3 — Neural-Network Layers

Build reusable neural-network components:

* dense/linear layers
* ReLU and other activation functions
* forward passes
* backward passes
* parameter gradients

### Stage 4 — Classification

Implement the machinery required for multiclass classification:

* logits
* softmax
* cross-entropy loss
* mini-batches
* stochastic gradient descent

### Stage 5 — MNIST

Load the MNIST handwritten-digit dataset directly from its binary files and train a small neural network.

An initial architecture might be:

```text
28 x 28 image
     |
     v
784 inputs
     |
     v
128 hidden units
     |
     v
10 outputs
     |
     v
digits 0-9
```

This stage will provide a complete example of training and evaluating a neural network without a machine-learning framework.

### Stage 6 — Training Improvements

After the basic network works correctly, investigate improvements such as:

* better parameter initialization
* mini-batch training
* Adam
* normalization
* model serialization
* profiling
* numerical stability
* performance optimization

These features will be introduced only after the simpler implementation is understood.

### Stage 7 — Toward a Language Model

The mathematical and software foundations developed for MNIST will then be reused to build an actual small language model.

Topics will include:

* tokenization
* embeddings
* positional information
* self-attention
* query, key, and value projections
* causal masking
* multi-head attention
* transformer blocks
* language-model cross-entropy loss
* text generation and sampling

The eventual goal is a small transformer whose important computations can be traced all the way down to C loops and floating-point arithmetic that we implemented ourselves.

## Project Structure

The repository will start small and grow as abstractions become necessary.

A likely eventual structure is:

```text
.
├── Makefile
├── README.md
├── .gitignore
│
├── include/
│   ├── matrix.h
│   ├── random.h
│   ├── layers.h
│   ├── loss.h
│   └── mnist.h
│
├── src/
│   ├── matrix.c
│   ├── random.c
│   ├── layers.c
│   ├── loss.c
│   └── mnist.c
│
├── examples/
│   ├── 01_neuron.c
│   ├── 02_matrix.c
│   ├── 03_dense.c
│   ├── 04_classifier.c
│   └── 05_mnist.c
│
├── tests/
│
├── data/
│
└── build/
```

Not all of these files should be created immediately.

At the beginning, the repository may contain little more than:

```text
.
├── Makefile
├── README.md
├── .gitignore
├── examples/
│   └── 01_neuron.c
├── tests/
└── build/
```

New modules should appear when the code demonstrates a need for them.

## Building

During development, compilation should favor diagnostics over performance.

Typical compiler options:

```text
-std=c17
-Wall
-Wextra
-Wpedantic
-Wshadow
-Wconversion
-O0
-g3
```

For memory and undefined-behavior debugging, builds may additionally use:

```text
-fsanitize=address,undefined
```

Optimized builds will be introduced later when performance becomes relevant.

## Testing

Correctness is particularly important when implementing backpropagation manually.

Where possible, analytical gradients will be checked against numerical gradients.

For a parameter \(w\), the analytical derivative

```text
dL/dw
```

can be compared with the finite-difference approximation:

```text
             L(w + epsilon) - L(w - epsilon)
dL/dw ~=    ---------------------------------
                         2 * epsilon
```

This technique will help distinguish mathematical errors from ordinary C programming errors.

The project will initially use a small custom test harness rather than an external testing framework.

## What This Project Is Not

This is not intended to compete with production machine-learning frameworks.

Libraries such as PyTorch provide highly optimized tensor operations, automatic differentiation, GPU execution, distributed training, and many other capabilities.

Here, those abstractions are intentionally avoided so that we can examine the mechanisms they normally hide.

The question guiding the project is not:

> How quickly can we train a neural network?

It is:

> Can we explain exactly what every important operation is doing, mathematically and in C?

## Current Status

**Module 1: Single Neuron — next**

The first implementation will train a neuron with two inputs.

We will derive and implement:

```text
z = w0*x0 + w1*x1 + b
```

followed by a loss function and gradient descent.

No matrix library, neural-network framework, or automatic differentiation is required.

From there, we build upward one abstraction at a time.
