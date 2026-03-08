# ∫ Calculus for Machine Learning

> *"To understand gradient descent, you must first understand gradients. To understand gradients, you must understand derivatives. To understand derivatives, you must understand limits. But practically — you just need to understand: which way is downhill?"*

Calculus is fundamentally about one thing: **how does a function change when its inputs change?** The derivative captures this precisely. Once you understand derivatives, gradients follow naturally — and with gradients, you can find the minimum of almost any function.

---

## Table of Contents

1. [Functions, Limits & Continuity](#1-functions-limits--continuity)
2. [Derivatives — Rate of Change](#2-derivatives--rate-of-change)
3. [Derivative Rules](#3-derivative-rules)
4. [The Chain Rule — Heart of Backpropagation](#4-the-chain-rule--heart-of-backpropagation)
5. [Partial Derivatives](#5-partial-derivatives)
6. [Gradients — Derivatives in Multiple Dimensions](#6-gradients--derivatives-in-multiple-dimensions)
7. [The Jacobian](#7-the-jacobian)
8. [The Hessian](#8-the-hessian)
9. [Gradient Descent — From Scratch](#9-gradient-descent--from-scratch)
10. [Loss Landscapes — Local Minima, Saddle Points & Convexity](#10-loss-landscapes--local-minima-saddle-points--convexity)
11. [Backpropagation — The Chain Rule in Action](#11-backpropagation--the-chain-rule-in-action)
12. [Advanced Optimisers](#12-advanced-optimisers)

---

## 1. Functions, Limits & Continuity

### What is a Function?

A function $f: \mathbb{R}^n \to \mathbb{R}^m$ maps inputs to outputs. Common examples:

- $f(x) = x^2$ maps a number to its square
- $f(x, y) = x^2 + y^2$ maps a 2D point to a scalar
- $f(\mathbf{x}) = \mathbf{Ax}$ maps a vector to another vector via matrix multiplication

### The Limit — Formalising "Getting Close"

$$\lim_{x \to a} f(x) = L$$

Means: as $x$ gets arbitrarily close to $a$, $f(x)$ gets arbitrarily close to $L$.

### Continuity

$f$ is **continuous** at $a$ if $\lim_{x \to a} f(x) = f(a)$ — no jumps or holes in the graph.

---

## 2. Derivatives — Rate of Change

### Definition from First Principles

The derivative $f'(a)$ is the slope of $f$ at $a$:

$$f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}$$

Alternative notation: $\frac{df}{dx}\bigg|_{x=a}$, or simply $\dot{f}(a)$.

**Geometric meaning**: Slope of the tangent line to $f$ at $a$.

**Physical meaning**: Rate of change of $f$ w.r.t. $x$ at $a$.

### Common Derivatives

| Function $f(x)$ | Derivative $f'(x)$ |
|-----------------|-------------------|
| $c$ (constant) | $0$ |
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln(x)$ | $1/x$ |
| $\sin(x)$ | $\cos(x)$ |
| $\cos(x)$ | $-\sin(x)$ |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ |

The sigmoid derivative is elegant: $\sigma'(x) = \sigma(x)(1-\sigma(x))$. If you already know $\sigma(x)$, the derivative costs almost nothing to compute.

---

## 3. Derivative Rules

### Power Rule

$$\frac{d}{dx} x^n = n x^{n-1}$$

### Sum Rule

$$\frac{d}{dx}[f(x) + g(x)] = f'(x) + g'(x)$$

### Product Rule

$$\frac{d}{dx}[f(x) \cdot g(x)] = f'(x) g(x) + f(x) g'(x)$$

### Quotient Rule

$$\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{f'(x)g(x) - f(x)g'(x)}{[g(x)]^2}$$

### Chain Rule ← Most Important

$$\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$$

Or with $u = g(x)$:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

---

## 4. The Chain Rule — Heart of Backpropagation

The chain rule computes derivatives of **composed functions**. Neural networks are compositions of functions, so the chain rule is literally what makes backpropagation work.

### Extended Chain Rule

For $y = f_n(f_{n-1}(\cdots f_1(x) \cdots))$:

$$\frac{dy}{dx} = \frac{df_n}{df_{n-1}} \cdot \frac{df_{n-1}}{df_{n-2}} \cdots \frac{df_1}{dx}$$

### Example: 2-Layer Network

$$z_1 = \mathbf{w}_1^T \mathbf{x}, \quad a_1 = \sigma(z_1), \quad z_2 = w_2 a_1, \quad \mathcal{L} = (y - z_2)^2$$

By chain rule:

$$\frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}$$

$$= 2(z_2 - y) \cdot w_2 \cdot \sigma'(z_1) \cdot x$$

The gradient of the loss w.r.t. a parameter deep in the network = **product of all downstream derivatives**. This is backpropagation.

---

## 5. Partial Derivatives

When $f$ depends on multiple variables, the **partial derivative** w.r.t. $x_i$ treats all other variables as constants:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_n)}{h}$$

### Example

For $f(x, y) = x^2 + 3xy + y^2$:

$$\frac{\partial f}{\partial x} = 2x + 3y \quad (\text{treat } y \text{ as constant})$$
$$\frac{\partial f}{\partial y} = 3x + 2y \quad (\text{treat } x \text{ as constant})$$

---

## 6. Gradients — Derivatives in Multiple Dimensions

The **gradient** of $f: \mathbb{R}^n \to \mathbb{R}$ is the vector of all partial derivatives:

$$\nabla_\mathbf{x} f = \begin{pmatrix} \partial f / \partial x_1 \\ \partial f / \partial x_2 \\ \vdots \\ \partial f / \partial x_n \end{pmatrix} \in \mathbb{R}^n$$

### Key Properties

- The gradient **points in the direction of steepest ascent**
- Moving **against** the gradient ($-\nabla f$) is the direction of steepest descent
- The gradient is **perpendicular to the level curves** of $f$
- At a minimum: $\nabla f(\mathbf{x}^*) = \mathbf{0}$

### The Directional Derivative

The rate of change of $f$ in direction $\mathbf{u}$ (unit vector):

$$D_\mathbf{u} f = \nabla f \cdot \mathbf{u} = \|\nabla f\| \cos\theta$$

Maximum when $\mathbf{u} = \frac{\nabla f}{\|\nabla f\|}$ — confirming the gradient is the steepest direction.

The negative gradient $-\nabla f$ points toward the steepest decrease. This is the geometric reason why **gradient descent** works — by taking small steps in that direction, you walk downhill on the function surface.

---

## 7. The Jacobian

For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ (vector-valued function), the **Jacobian** is the matrix of all partial derivatives:

$$\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{pmatrix} \partial f_1/\partial x_1 & \cdots & \partial f_1/\partial x_n \\ \vdots & \ddots & \vdots \\ \partial f_m/\partial x_1 & \cdots & \partial f_m/\partial x_n \end{pmatrix} \in \mathbb{R}^{m \times n}$$

The gradient is a special case: when $m=1$, $\mathbf{J} = \nabla f^T$. The Jacobian generalises this to vector-valued functions.

---

## 8. The Hessian

For $f: \mathbb{R}^n \to \mathbb{R}$, the **Hessian** is the matrix of second derivatives:

$$\mathbf{H}_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

$$\mathbf{H} = \nabla^2 f = \begin{pmatrix} \partial^2 f/\partial x_1^2 & \partial^2 f/\partial x_1 \partial x_2 & \cdots \\ \partial^2 f/\partial x_2 \partial x_1 & \partial^2 f/\partial x_2^2 & \cdots \\ \vdots & & \ddots \end{pmatrix}$$

The Hessian is **symmetric** ($H_{ij} = H_{ji}$ by Schwarz's theorem).

### What the Hessian Tells You

At a critical point ($\nabla f = 0$):
- All eigenvalues of $\mathbf{H}$ positive → **local minimum**
- All eigenvalues negative → **local maximum**
- Mixed signs → **saddle point**

The **condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$ measures curvature anisotropy. Large $\kappa$ → elongated loss bowl → slow gradient descent convergence.

**Newton's method** uses the Hessian to take curvature-aware steps:

$$\theta \leftarrow \theta - \mathbf{H}^{-1} \nabla f$$

This converges much faster than gradient descent for smooth functions. The downside is inverting $\mathbf{H}$ costs $O(n^3)$, which is infeasible for large problems.

---

## 9. Gradient Descent — From Scratch

The fundamental parameter update rule:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

Where $\eta$ is the **learning rate** — controls step size.

### Variants

| Variant | Update | Batch Size |
|---------|--------|-----------|
| Batch GD | $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$ | Entire dataset |
| Stochastic GD (SGD) | $\theta \leftarrow \theta - \eta \nabla_\theta \ell(\theta; x_i, y_i)$ | 1 sample |
| Mini-batch SGD | $\theta \leftarrow \theta - \eta \frac{1}{B}\sum_{i \in \mathcal{B}} \nabla_\theta \ell_i$ | Batch of $B$ |

### SGD with Momentum

```
v_t = β v_{t-1} + (1-β) ∇L(θ_t)
θ_{t+1} = θ_t - η v_t
```

Momentum accumulates gradients in consistent directions, dampens oscillations.

### Adam (Adaptive Moment Estimation)

Keeps running estimates of first moment (mean gradient) $m_t$ and second moment (mean squared gradient) $v_t$:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1-\beta_1^t), \quad \hat{v}_t = v_t / (1-\beta_2^t) \quad \text{(bias correction)}$$
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Adam adapts the learning rate **per parameter** — parameters with large gradient history get smaller effective learning rates.

---

## 10. Loss Landscapes — Local Minima, Saddle Points & Convexity

### Convex vs Non-Convex

A function $f$ is **convex** if for all $\mathbf{x}, \mathbf{y}$ and $\lambda \in [0,1]$:

$$f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y})$$

- **Convex**: Every local minimum is a global minimum — gradient descent is guaranteed to find it
- **Non-convex**: Multiple local minima and saddle points may exist

### Types of Critical Points (where $\nabla f = 0$)

| Type | Gradient | Hessian Eigenvalues |
|------|----------|---------------------|
| Global minimum | $\mathbf{0}$ | All positive |
| Local minimum | $\mathbf{0}$ | All positive |
| Saddle point | $\mathbf{0}$ | Mixed signs |
| Maximum | $\mathbf{0}$ | All negative |

---

## 11. Backpropagation — The Chain Rule in Action

Backprop is just the chain rule applied systematically through a computational graph. For each layer:

**Forward pass**: Compute and cache intermediate values
**Backward pass**: Propagate gradients from output to input using the chain rule

### The Four Fundamental Backprop Equations (Neural Network)

For layer $l$ with $\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$ and $\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})$:

$$\boldsymbol{\delta}^{(L)} = \nabla_{\mathbf{a}^{(L)}} \mathcal{L} \odot \sigma'(\mathbf{z}^{(L)}) \quad \text{(output layer error)}$$
$$\boldsymbol{\delta}^{(l)} = \left((\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)}\right) \odot \sigma'(\mathbf{z}^{(l)}) \quad \text{(hidden layer error)}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T \quad \text{(weight gradient)}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)} \quad \text{(bias gradient)}$$

Where $\odot$ is elementwise multiplication (Hadamard product).

The error signal $\boldsymbol{\delta}^{(l)}$ propagates **backward** through the network, multiplied by the transpose weight matrices $(\mathbf{W}^{(l)})^T$ at each step.

---

## 12. Advanced Optimisers

### Learning Rate Schedules

- **Step decay**: Multiply by $\gamma$ every $k$ epochs
- **Cosine annealing**: $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t/T))$
- **Warmup**: Start with a small learning rate, ramp it up, then decay

### Why Warmup?

At the start of training, gradients are noisy and parameter estimates are poor. A large initial learning rate can cause the optimisation to overshoot badly. Starting with a small rate and increasing it gradually gives the algorithm time to settle before taking larger steps.

### Gradient Clipping

$$\text{if } \|\mathbf{g}\|_2 > \tau: \quad \mathbf{g} \leftarrow \frac{\tau}{\|\mathbf{g}\|_2} \mathbf{g}$$

If the gradient becomes very large, it is rescaled to have norm exactly $\tau$. This prevents a single bad update from ruining the optimisation.

---

## 📓 Notebook

Open [`calculus.ipynb`](./calculus.ipynb) for hands-on derivations, gradient descent from scratch, backpropagation implementation, and loss landscape visualisations.
