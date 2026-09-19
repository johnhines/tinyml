Yes. That’s a good project for learning the machinery rather than hiding it behind a framework.

One terminology point first: **NIST/MNIST handwritten-digit recognition is a classification problem, not inherently an LLM problem.** An LLM normally predicts tokens in sequences. But MNIST is an excellent first target because we can build the same fundamental machinery—tensors, linear layers, nonlinearities, losses, backpropagation, optimization—in a setting where debugging is much easier. Once that works, we can reuse most of the infrastructure to build a tiny transformer language model.

I suggest we proceed in small stages and **not use BLAS, PyTorch, TensorFlow, NumPy, or neural-network libraries**. C standard library plus `libm` should be enough initially. We can even write our own matrix multiplication and random initialization.

A sensible progression is:

1. **Scalar neuron** — implement one artificial neuron in C and derive forward propagation, loss, gradients, and gradient descent by hand.
2. **Vectors and matrices** — build a tiny tensor/matrix module: allocation, indexing, dot products, matrix-vector multiplication, matrix multiplication.
3. **Dense layer** — implement \(y=Wx+b\), including its backward pass.
4. **Activations** — ReLU initially, with derivatives.
5. **Classification** — implement softmax and cross-entropy ourselves.
6. **MNIST loader** — parse the binary MNIST/NIST-format image and label files directly in C.
7. **First network** — something deliberately simple such as \(784\rightarrow128\rightarrow10\).
8. **Training** — mini-batches, SGD, initialization, shuffling, epochs, and validation.
9. **Debugging the math** — numerical gradient checking so we know our backward implementation is actually correct.
10. **Improve it** — Adam, better initialization, normalization, serialization, inference, etc.
11. **Then turn toward an actual tiny LLM** — embeddings, tokenization, attention, transformer blocks, causal masking, language-model loss, sampling.

The central idea we'll encounter immediately is very simple. A neuron computes

$$
z=\sum_{i=1}^{n}w_i x_i+b
$$

followed by some activation

$$
y=f(z).
$$

Training means finding values of \(w_i\) and \(b\) that reduce a loss \(L\). Gradient descent performs

$$
w_i \leftarrow w_i-\eta\frac{\partial L}{\partial w_i},
$$

where \(\eta\) is the learning rate. **Backpropagation is essentially an efficient application of the chain rule for calculating those derivatives.**

I'd like to resist jumping directly to the 784×128 network. Our **first module should be a single neuron with perhaps two inputs**, trained on four or so examples. We'll write the forward calculation explicitly, derive every derivative on paper, implement the backward calculation explicitly in C, and watch gradient descent actually change the weights.

Once that tiny program makes complete sense, we'll generalize it into matrices rather than introducing abstractions before we understand what they're abstracting.

So our first milestone can be just:

```text
two inputs
    │
    ├── w0 ──┐
    │        │
    └── w1 ──┼──> weighted sum + bias ──> activation ──> prediction
             │
             └──> loss ──> derivatives ──> update w0,w1,b
```

And we'll implement **every operation behind that diagram ourselves in C**.

When you're ready, say **“start module 1”**, and we'll build the single-neuron program slowly, beginning with what a weight actually represents and deriving its gradient before writing much code.

Good. Before we let the neuron train repeatedly, we're going to do something important: **prove that the gradients we coded are probably correct.**

# Module 1.2 — Numerical Gradient Checking

Our backward pass says:

$$
\frac{\partial L}{\partial w_0}
=
2(\hat y-y)x_0
$$

But that's a formula we derived and typed ourselves. Either step could contain a mistake.

We need an independent way to check it.

## 1. What does a derivative mean?

For an ordinary function

$$
f(x),
$$

the derivative at \(x\) describes how much \(f\) changes when we make a tiny change to \(x\).

One approximation is

$$
f'(x)\approx\frac{f(x+\epsilon)-f(x)}{\epsilon}.
$$

A better approximation evaluates both sides:

$$
\boxed{
f'(x)\approx
\frac{f(x+\epsilon)-f(x-\epsilon)}
{2\epsilon}
}
$$

This is called a **central finite difference**.

The smaller \(\epsilon\) is, within numerical limits, the more closely we're examining the local slope.

For example, consider

$$
f(x)=x^2.
$$

At \(x=3\), calculus tells us

$$
f'(3)=2(3)=6.
$$

Take \(\epsilon=0.001\):

$$
\frac{(3.001)^2-(2.999)^2}{0.002}
\approx6.
$$

No symbolic differentiation was required.

That's exactly what we'll do with our neural network.

---

# 2. Treat loss as a function of a weight

Keep everything fixed except \(w_0\).

Conceptually, our loss becomes

$$
L(w_0).
$$

We can calculate it three times:

```text id="ij6ixf"
                  loss
                   ^
                   |
              .----
           .-'
        .-'
-------+------------------> w0
     w0-e   w0   w0+e
```

Specifically:

$$
L_+=L(w_0+\epsilon)
$$

and

$$
L_-=L(w_0-\epsilon).
$$

Then

$$
\frac{\partial L}{\partial w_0}
\approx
\frac{L_+-L_-}{2\epsilon}.
$$

This calculation knows absolutely nothing about backpropagation.

That's why it's useful.

If our backpropagation says

$$
\frac{\partial L}{\partial w_0}=14.4
$$

and finite differences independently say

$$
\frac{\partial L}{\partial w_0}\approx14.400000,
$$

that's strong evidence that our derivative is right.

---

# 3. First small refactoring

We currently have the forward computation directly inside `main()`.

We're going to extract the loss calculation into a function because numerical gradient checking needs to evaluate it repeatedly.

Add this above `main()`:

```c id="ql46cp"
static double compute_loss(
    double x0,
    double x1,
    double y,
    double w0,
    double w1,
    double b)
{
    double y_hat = w0 * x0 + w1 * x1 + b;
    double error = y_hat - y;

    return error * error;
}
```

Notice what we're **not** doing.

We're not creating:

```c id="5kph3r"
typedef struct {
    ...
} Neuron;
```

We're not creating tensors or a `Layer` abstraction.

We're simply extracting one mathematical function:

$$
L(x_0,x_1,y,w_0,w_1,b).
$$

That's enough.

---

# 4. Numerically differentiate \(w_0\)

Before your gradient-descent update, add:

```c id="dgrd4k"
double epsilon = 1e-5;

double loss_plus = compute_loss(
    x0, x1, y,
    w0 + epsilon, w1, b);

double loss_minus = compute_loss(
    x0, x1, y,
    w0 - epsilon, w1, b);

double numerical_dw0 =
    (loss_plus - loss_minus) / (2.0 * epsilon);

printf("\ngradient check:\n");
printf("analytical dL/dw0 = %.10f\n", dL_dw0);
printf("numerical  dL/dw0 = %.10f\n", numerical_dw0);
```

You should see values close to:

```text id="ibmgi6"
analytical dL/dw0 = 14.4000000000
numerical  dL/dw0 = 14.4000000000
```

They might not be *exactly* identical because floating-point arithmetic has finite precision.

---

# 5. Why does this work?

Let's inspect what we're actually doing.

Our original weight is

$$
w_0=0.5.
$$

We ask:

> What happens if I make \(w_0\) just slightly larger?

Then:

$$
w_0+\epsilon=0.50001.
$$

We calculate the loss.

Next:

> What happens if I make \(w_0\) just slightly smaller?

$$
w_0-\epsilon=0.49999.
$$

We calculate the loss again.

If increasing \(w_0\) makes the loss increase, while decreasing \(w_0\) makes it decrease, then the slope is positive:

```text id="aqq6jm"
loss
 ^
 |                /
 |              /
 |            /
 |          *
 |        /
 |      /
 +--------------------> w0
          0.5

       positive slope
```

Our derivative is approximately \(+14.4\).

That's why gradient descent performs

$$
w_0\leftarrow
w_0-\eta(14.4).
$$

It moves \(w_0\) **left**, toward smaller values.

---

# 6. Check all three parameters

Now I want you to implement the checks for \(w_1\) and \(b\) yourself.

For \(w_1\), compute:

$$
\frac{
L(w_1+\epsilon)-L(w_1-\epsilon)
}{
2\epsilon
}.
$$

Everything except \(w_1\) stays unchanged.

For \(b\):

$$
\frac{
L(b+\epsilon)-L(b-\epsilon)
}{
2\epsilon
}.
$$

Your output should end up approximately:

```text id="t2yjdv"
gradient check:

parameter     analytical       numerical
w0            14.400000        14.400000
w1            21.600000        21.600000
b              7.200000         7.200000
```

Don't worry about making an actual formatted table unless you want to.

---

# 7. A subtle but important experiment

After everything agrees, deliberately break the backward pass.

For example, change:

```c id="zkxjq8"
double dL_dw0 = dL_dy_hat * x0;
```

to:

```c id="i2bx7a"
double dL_dw0 = dL_dy_hat;
```

Now run the program.

You should get something like:

```text id="guy01c"
analytical dL/dw0 = 7.200000
numerical  dL/dw0 = 14.400000
```

Our gradient checker has caught the bug.

Then restore the correct calculation.

This distinction will become extremely important later:

```text id="e2nixn"
              BACKPROP
           (our calculus)
                 |
                 v
            dL/dw = 14.4
                 |
                 | compare
                 |
            dL/dw ~ 14.4
                 ^
                 |
        FINITE DIFFERENCES
     (numerical measurement)
```

If both implementations contain different logic but produce the same answer, we gain confidence in the gradient.

## Why we're doing this so early

When we eventually have

$$
784\rightarrow128\rightarrow10,
$$

there will be roughly \(100{,}000\) parameters.

And when we reach attention, we'll encounter chains such as

$$
X
\rightarrow Q,K,V
\rightarrow QK^T
\rightarrow \text{scale}
\rightarrow \text{mask}
\rightarrow \operatorname{softmax}
\rightarrow AV
\rightarrow L.
$$

A tiny mistake in one backward operation can allow the program to compile and run perfectly while making the model incapable of learning.

So numerical gradient checking is going into our toolbox now.

### Your task before Module 1.3

Extend `01_neuron.c` so that the numerical gradients for **all three** parameters agree with the analytical gradients. Also perform the deliberate-bug experiment once.

After that, we'll move to the fun part: **give the neuron multiple examples, introduce epochs, and watch gradient descent actually recover the hidden relationship**

$$
\boxed{y=2x_0-3x_1+1}.
$$

Excellent. That means the analytical backpropagation and the independent numerical calculation agree at the same parameter values.

There's an important lesson from the bug you just found: **a gradient only describes the slope at one particular point in parameter space.** Once we update the parameters, we're at a new point, and the gradients generally change.

## Module 1.3 — Learning from multiple examples

Now let's make the neuron actually learn the hidden function

$$
\boxed{y=2x_0-3x_1+1}.
$$

We'll give it several examples generated from that rule:

| \(x_0\) | \(x_1\) | \(y\) |
| ------: | ------: | ----: |
|       0 |       0 |     1 |
|       1 |       0 |     3 |
|       0 |       1 |    -2 |
|       1 |       1 |     0 |
|       2 |       1 |     2 |
|       1 |       2 |    -3 |

The neuron doesn't know the rule. It sees only those input/output pairs.

### Representing our dataset

For now, let's resist making a dataset abstraction. Add three arrays:

```c
double x0_data[] = {0.0, 1.0, 0.0, 1.0, 2.0, 1.0};
double x1_data[] = {0.0, 0.0, 1.0, 1.0, 1.0, 2.0};
double y_data[]  = {1.0, 3.0, -2.0, 0.0, 2.0, -3.0};

size_t n = 6;
```

You'll need:

```c
#include <stddef.h>
```

although `stdio.h` commonly brings in definitions indirectly; it's better not to depend on that.

Now training consists of repeatedly visiting these examples.

## One training example

For example \(i\), the forward pass is:

```c
double x0 = x0_data[i];
double x1 = x1_data[i];
double y  = y_data[i];

double y_hat = w0 * x0 + w1 * x1 + b;
double error = y_hat - y;
double loss = error * error;
```

And our gradients remain exactly what we derived:

```c
double dL_dy_hat = 2.0 * error;

double dL_dw0 = dL_dy_hat * x0;
double dL_dw1 = dL_dy_hat * x1;
double dL_db  = dL_dy_hat;
```

Then:

```c
w0 -= learning_rate * dL_dw0;
w1 -= learning_rate * dL_dw1;
b  -= learning_rate * dL_db;
```

Nothing about the calculus changed just because we now have multiple examples.

## Introducing an epoch

One complete trip through the training dataset is called an **epoch**.

So our loop structure becomes:

```c
for (size_t epoch = 0; epoch < epochs; ++epoch) {

    for (size_t i = 0; i < n; ++i) {

        /* forward */

        /* backward */

        /* update */
    }
}
```

This is our first real training loop.

Set:

```c
size_t epochs = 1000;
double learning_rate = 0.01;
```

Don't worry yet about whether 1000 or 0.01 are "correct." Learning rate selection is something we'll investigate.

### Track the loss

There's one more useful addition.

At the beginning of each epoch:

```c
double total_loss = 0.0;
```

For every example:

```c
total_loss += loss;
```

Then calculate the average:

```c
double mean_loss = total_loss / (double)n;
```

Print it periodically rather than 1000 times:

```c
if (epoch % 100 == 0) {
    printf("epoch %zu: loss = %.10f\n",
           epoch, mean_loss);
}
```

One subtlety: because we're updating the weights after every example, this `mean_loss` combines losses evaluated at slightly different parameter values during the epoch. That's perfectly adequate for watching training right now. Later we'll be more precise about training metrics.

## What kind of gradient descent did we just implement?

Notice this:

```text
example 0 -> gradient -> UPDATE
example 1 -> gradient -> UPDATE
example 2 -> gradient -> UPDATE
...
```

We're updating after **each individual example**.

That's stochastic gradient descent, or more precisely single-example SGD in our current setup.

Another possibility would be:

```text
example 0 -> gradient --\
example 1 -> gradient ---\
example 2 -> gradient ----+-> average gradients -> ONE UPDATE
example 3 -> gradient ---/
...
```

That's batch gradient descent if we use the entire dataset.

And later:

```text
32 examples -> average gradients -> update
32 examples -> average gradients -> update
32 examples -> average gradients -> update
```

is **mini-batch gradient descent**, which is what we'll eventually want.

But don't implement batches yet.

## What should happen?

After training, print:

```c
printf("\nlearned parameters:\n");
printf("w0 = %.6f\n", w0);
printf("w1 = %.6f\n", w1);
printf("b  = %.6f\n", b);
```

Remember the hidden function was

$$
y=2x_0-3x_1+1.
$$

Therefore, we'd like training to discover approximately

$$
w_0\approx2
$$

$$
w_1\approx-3
$$

$$
b\approx1.
$$

That correspondence is worth appreciating:

$$
\underbrace{y=2x_0-3x_1+1}_{\text{function that generated our data}}
$$

versus

$$
\underbrace{\hat y=w_0x_0+w_1x_1+b}_{\text{model}}
$$

Training is searching for parameters that make the second expression reproduce the first.

### One experiment I want you to do

After it works, change the initial parameters substantially. For example:

```c
double w0 = -4.0;
double w1 =  2.5;
double b  = -1.0;
```

Run it again.

Then try:

```c
double w0 = 10.0;
double w1 = -8.0;
double b  = 5.0;
```

Observe whether the parameters still approach \(2,-3,1\), and watch how the loss behaves.

Also try changing:

```c
double learning_rate = 0.01;
```

to values such as:

```text
0.001
0.1
1.0
```

Do these one at a time.

**Pay particular attention to `1.0`.** Don't assume a larger learning rate means faster learning. If you see enormous values, `inf`, or `nan`, that's an important result rather than a failed experiment.

Once you've run those experiments, tell me what happened—especially with the different learning rates. From those observations we'll develop an intuitive and mathematical picture of **why gradient descent converges, oscillates, or explodes** before moving beyond our single neuron.

