# MNIST Digit Classifier — Neural Network from Scratch (NumPy)

A fully-connected neural network built **from scratch using only NumPy** (no PyTorch/TensorFlow) that classifies handwritten digits (0–9) from the MNIST dataset. Every operation — forward pass, backpropagation, and the Adam optimizer — is implemented manually with matrix math.

**Test Accuracy: 97.53%** | **Final Train Accuracy: 99.52%**

---

## 1. Architecture

A 3-layer feedforward network:

```
Input (784)  →  Hidden 1 (128, ReLU)  →  Hidden 2 (128, ReLU)  →  Output (10, Softmax)
```

| Layer | Shape | Activation | Init |
|---|---|---|---|
| Input | 784 (28×28 flattened pixels) | – | – |
| Hidden 1 | 128 | ReLU | He |
| Hidden 2 | 128 | ReLU | He |
| Output | 10 (digit classes) | Softmax | Xavier |

- **He initialization** (`W ~ N(0, 2/n_in)`) is used for the ReLU layers to keep activation variance stable across depth.
- **Xavier initialization** (`W ~ N(0, 1/n_in)`) is used for the output layer, which pairs naturally with softmax.

---

## 2. Forward Pass

For each layer, a linear transform is followed by an activation:

$$z^{[l]} = h^{[l-1]} W^{[l]} + b^{[l]}$$

**Hidden layers (ReLU):**
$$h^{[l]} = \text{ReLU}(z^{[l]}) = \max(0, z^{[l]})$$

**Output layer (Softmax)** — converts raw scores into class probabilities that sum to 1:
$$\hat{y}_i = \text{softmax}(z^{[3]})_i = \frac{e^{z_i}}{\sum_{k=1}^{10} e^{z_k}}$$

The code subtracts `max(z)` before exponentiating (`np.exp(z - np.max(z))`) purely for numerical stability — it doesn't change the result since it cancels out in the ratio.

---

## 3. Loss Function — Categorical Cross-Entropy

Labels are one-hot encoded (e.g. digit `5` → `[0,0,0,0,0,1,0,0,0,0]`). For a batch of size *m*:

$$\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{10} y_{i,k}\log(\hat{y}_{i,k})$$

Since `y` is one-hot, this reduces to `-log(predicted probability of the correct class)`, averaged over the batch. Predictions are clipped to `[1e-15, 1-1e-15]` to avoid `log(0)`.

---

## 4. Backward Pass (Backpropagation)

The gradients are derived layer-by-layer using the chain rule, propagating the error backward from the output.

**Output layer.** The beautiful result of pairing softmax with cross-entropy is that the gradient at the output simplifies to just the residual:
$$dz^{[3]} = \hat{y} - y$$

**Weight/bias gradients** at any layer *l* (with `hl` = the previous layer's activated output):
$$dW^{[l]} = \frac{1}{m}(h^{[l-1]})^T \, dz^{[l]} \qquad db^{[l]} = \frac{1}{m}\sum dz^{[l]}$$

**Propagating error to earlier layers** — the upstream gradient is projected back through the weights, then masked by the ReLU derivative (1 where `z > 0`, else 0):
$$dz^{[l]} = \left(dz^{[l+1]} (W^{[l+1]})^T\right) \odot \text{ReLU}'(z^{[l]})$$

This is applied twice — from output→hidden2, and hidden2→hidden1 — giving `dw1, db1, dw2, db2, dw3, db3`.

---

## 5. Optimizer — Adam

Instead of plain gradient descent, the model uses **Adam** (Adaptive Moment Estimation), which tracks a running average of both the gradient (momentum) and the squared gradient (adaptive learning rate) for every parameter.

For each parameter, at step *t*:

**1st moment (mean of gradients):**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\,dW$$

**2nd moment (uncentered variance of gradients):**
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)\,dW^2$$

**Bias correction** (moments start at 0, so early steps are corrected upward):
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

**Parameter update:**
$$W \leftarrow W - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

With $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, and learning rate $\eta = 0.001$. This is applied independently to `w1, b1, w2, b2, w3, b3`.

---

## 6. Training Setup

| Hyperparameter | Value |
|---|---|
| Epochs | 12 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | Cross-entropy |

**Pipeline:**
1. Load MNIST (70,000 images, 28×28) via `sklearn.datasets.fetch_openml`.
2. Normalize pixels to `[0, 1]` (`/255.0`) and one-hot encode labels.
3. Split: 60,000 train / 10,000 test.
4. Each epoch: shuffle data → split into mini-batches of 64 → forward pass → compute loss → backward pass → Adam update.
5. Track loss/accuracy per epoch; evaluate on the held-out test set; visualize with a confusion matrix.

---

## 7. Results

| Epoch | Loss | Train Accuracy |
|---|---|---|
| 1 | 0.268 | 92.22% |
| 4 | 0.055 | 98.32% |
| 8 | 0.023 | 99.28% |
| 12 | 0.014 | **99.52%** |

**Final Test Accuracy: 97.53%**

The ~2-point gap between train and test accuracy is normal — it reflects mild overfitting on a model with no regularization (no dropout/weight decay), which is expected for a "from scratch" educational build.

---

## 8. Project Structure

```
mnist_scratch.ipynb    # Full notebook: data loading → model → training → evaluation
```

The core is a single `NeuralNet` class implementing `forward_pass`, `compute_loss`, and `backward_pass` (which also performs the Adam update) — no autograd, no deep learning framework, just NumPy matrix operations.

## 9. Requirements

```
numpy
pandas
matplotlib
scikit-learn
```

## 10. Why Build From Scratch?

Frameworks like PyTorch hide backpropagation and optimizers behind `loss.backward()` and `optimizer.step()`. Implementing them manually here makes explicit:
- How gradients actually flow backward through a network (chain rule, layer by layer)
- Why softmax + cross-entropy pairs so cleanly (`dz = ŷ - y`)
- What Adam is really doing with momentum and adaptive step sizes, beyond "it just works"
