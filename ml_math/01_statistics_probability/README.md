# 📊 Statistics & Probability for Machine Learning

> *"All models are wrong, but some are useful."* — George Box

Statistics is not about memorizing formulas. It's about **reasoning under uncertainty** — the ability to draw conclusions from incomplete information.

---

## Table of Contents

1. [Populations & Sampling](#1-populations--sampling)
2. [Descriptive Statistics — Mean, Median, Mode](#2-descriptive-statistics--mean-median-mode)
3. [Expected Value](#3-expected-value)
4. [Variance & Standard Deviation](#4-variance--standard-deviation)
5. [Covariance & Correlation](#5-covariance--correlation)
6. [Random Variables](#6-random-variables)
7. [Probability Distributions](#7-probability-distributions)
8. [The Normal Distribution & Central Limit Theorem](#8-the-normal-distribution--central-limit-theorem)
9. [Conditional Probability & Bayes' Theorem](#9-conditional-probability--bayes-theorem)
10. [Maximum Likelihood Estimation (MLE)](#10-maximum-likelihood-estimation-mle)
11. [Linear Regression — The Statistical View](#11-linear-regression--the-statistical-view)
12. [Logistic Regression — Probability as Output](#12-logistic-regression--probability-as-output)

---

## 1. Populations & Sampling

### Intuition

You want to know the average height of all humans on Earth (~8 billion people). You can't measure everyone. So you **sample** — you pick a representative subset and use it to estimate the truth.

### Formal Definitions

- **Population**: The complete set of all entities you care about. Described by a true distribution $P$.
- **Sample**: A subset of $N$ observations drawn from the population: $\{x_1, x_2, \ldots, x_N\}$.
- **Statistic**: Any function of a sample (e.g., the sample mean $\bar{x}$).
- **Parameter**: The true value in the population (e.g., population mean $\mu$). Usually unknown.

The key tension: your sample is finite, but the population is vast. How accurately does what you measured reflect the true underlying reality?

---

## 2. Descriptive Statistics — Mean, Median, Mode

### Mean (Arithmetic Average)

$$\bar{x} = \frac{1}{N} \sum_{i=1}^{N} x_i$$

The mean **minimizes the sum of squared errors** — i.e., if you had to guess one value for all data points, the mean minimizes $\sum(x_i - \hat{x})^2$.

### Median

The **middle value** when data is sorted. Minimizes the **sum of absolute errors**: $\sum |x_i - \hat{x}|$. More robust to outliers than the mean.

### Mode

The most frequently occurring value. Useful when you need the single most common outcome.

---

## 3. Expected Value

### Intuition

The expected value $\mathbb{E}[X]$ is the long-run average if you repeated an experiment infinitely many times. It weights each possible outcome by its probability.

### Formula

For a **discrete** random variable:

$$\mathbb{E}[X] = \sum_{x} x \cdot P(X = x)$$

For a **continuous** random variable with density $p(x)$:

$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot p(x) \, dx$$

### Properties

$$\mathbb{E}[aX + b] = a\mathbb{E}[X] + b \quad \text{(linearity)}$$
$$\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y] \quad \text{(always true)}$$
$$\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y] \quad \text{(only if X, Y are independent)}$$

---

## 4. Variance & Standard Deviation

### Intuition

The mean tells you *where* your data is centered. Variance tells you *how spread out* it is.

### Population Variance

$$\sigma^2 = \mathbb{E}[(X - \mu)^2] = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$$

### Sample Variance (Bessel's Correction)

$$s^2 = \frac{1}{N-1} \sum_{i=1}^{N} (x_i - \bar{x})^2$$

We divide by $N-1$ instead of $N$ to get an **unbiased estimator** of the population variance. With $N$, we slightly underestimate spread.

### Standard Deviation

$$\sigma = \sqrt{\sigma^2}$$

Same units as the data — more interpretable than variance.



---

## 5. Covariance & Correlation

### Covariance

Covariance measures how two variables **move together**:

$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$$

- $\text{Cov}(X, Y) > 0$: When $X$ increases, $Y$ tends to increase
- $\text{Cov}(X, Y) < 0$: When $X$ increases, $Y$ tends to decrease
- $\text{Cov}(X, Y) = 0$: No linear relationship

### Correlation (Pearson)

Normalizes covariance to $[-1, 1]$:

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

### The Covariance Matrix

For a dataset with $d$ features, the covariance matrix $\Sigma$ is $d \times d$:

$$\Sigma_{ij} = \text{Cov}(X_i, X_j)$$

$$\Sigma = \frac{1}{N-1} (\mathbf{X} - \bar{\mathbf{X}})^T (\mathbf{X} - \bar{\mathbf{X}})$$



---

## 6. Random Variables

### Intuition

A **random variable** $X$ is a function that maps outcomes of a random experiment to numbers. It's not "random" in the sense of chaotic — it has a precise probability structure.

### Types

- **Discrete**: Takes countable values. E.g., number of words in a sentence, class label. Described by a **Probability Mass Function (PMF)**: $P(X = x)$
- **Continuous**: Takes any value in an interval. E.g., temperature, pixel intensity. Described by a **Probability Density Function (PDF)**: $p(x)$, where $P(a \leq X \leq b) = \int_a^b p(x) \, dx$

### The PDF Constraint

$$\int_{-\infty}^{\infty} p(x) \, dx = 1 \quad \text{(total probability = 1)}$$



---

## 7. Probability Distributions

### 7.1 Uniform Distribution

$$X \sim \text{Uniform}(a, b), \quad p(x) = \frac{1}{b-a} \text{ for } x \in [a,b]$$

All outcomes equally likely. A simple starting point.

### 7.2 Bernoulli Distribution

A single coin flip with probability $p$ of heads:

$$P(X=1) = p, \quad P(X=0) = 1-p$$
$$\mathbb{E}[X] = p, \quad \text{Var}(X) = p(1-p)$$

**Binary cross-entropy** is derived from the Bernoulli distribution.

### 7.3 Binomial Distribution

$n$ independent Bernoulli trials:

$$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$$

$$\mathbb{E}[X] = np, \quad \text{Var}(X) = np(1-p)$$

### 7.4 Normal (Gaussian) Distribution

The most important distribution in statistics:

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

$$X \sim \mathcal{N}(\mu, \sigma^2)$$

**Why it's everywhere:**
- The Central Limit Theorem makes it the natural limit of many averaging processes
- It is the maximum entropy distribution given only mean and variance constraints

### 7.5 Categorical Distribution

Generalization of Bernoulli to $K$ outcomes:

$$P(Y = k) = \pi_k, \quad \sum_{k=1}^K \pi_k = 1$$

The **softmax function** maps any real-valued vector $z$ to a valid categorical distribution:

$$\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

---

## 8. The Normal Distribution & Central Limit Theorem

### Why the Normal Distribution is Special

**Central Limit Theorem (CLT)**: If you take the average of $n$ independent, identically distributed (i.i.d.) random variables $X_1, \ldots, X_n$ with mean $\mu$ and variance $\sigma^2$, then as $n \to \infty$:

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{d} \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)$$

**Translation**: No matter what shape the original distribution is, averages of many samples become normally distributed. This is profound.

### The Standard Normal

$$Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$$

This **z-score** tells you how many standard deviations a data point is from the mean.

### 68-95-99.7 Rule

$$P(\mu - \sigma \leq X \leq \mu + \sigma) \approx 68\%$$
$$P(\mu - 2\sigma \leq X \leq \mu + 2\sigma) \approx 95\%$$
$$P(\mu - 3\sigma \leq X \leq \mu + 3\sigma) \approx 99.7\%$$



---

## 9. Conditional Probability & Bayes' Theorem

### Conditional Probability

The probability of event $A$ given that $B$ has already occurred:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

**Intuition**: You're restricting your universe to cases where $B$ happened, then asking how often $A$ also happens in that restricted universe.

### The Product Rule

$$P(A \cap B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$$

### Bayes' Theorem

Derived directly from the product rule — one of the most important equations in all of science:

$$\boxed{P(H \mid D) = \frac{P(D \mid H) \cdot P(H)}{P(D)}}$$

| Term | Name | Meaning |
|------|------|---------|
| $P(H \mid D)$ | **Posterior** | Probability of hypothesis given data |
| $P(D \mid H)$ | **Likelihood** | Probability of data given hypothesis |
| $P(H)$ | **Prior** | Initial belief in the hypothesis |
| $P(D)$ | **Evidence** | Normalizing constant |

### Law of Total Probability

$$P(D) = \sum_h P(D \mid H=h) \cdot P(H=h)$$



---

## 10. Maximum Likelihood Estimation (MLE)

### Intuition

Given observed data $\mathcal{D} = \{x_1, \ldots, x_N\}$ and a model with parameters $\theta$, MLE asks:

> **What values of $\theta$ make this data most probable?**

### The Likelihood Function

$$\mathcal{L}(\theta) = P(\mathcal{D} \mid \theta) = \prod_{i=1}^{N} p(x_i \mid \theta)$$

(Assumes i.i.d. data — each observation is independent.)

### Log-Likelihood (The Math Gets Nicer)

Because products are hard to optimize (numerically unstable), we take the log:

$$\ell(\theta) = \log \mathcal{L}(\theta) = \sum_{i=1}^{N} \log p(x_i \mid \theta)$$

Since $\log$ is monotone, maximizing $\ell(\theta)$ is equivalent to maximizing $\mathcal{L}(\theta)$.

### MLE → Familiar Loss Functions

**Case 1: Gaussian likelihood → MSE Loss**

If we model $y_i = f_\theta(x_i) + \epsilon_i$ where $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$:

$$\ell(\theta) = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i - f_\theta(x_i))^2$$

Maximizing log-likelihood = **minimizing MSE**. MSE is not arbitrary — it's MLE under Gaussian noise!

**Case 2: Bernoulli likelihood → Binary Cross-Entropy**

If $y_i \in \{0, 1\}$ and $P(y_i=1 \mid x_i) = \hat{p}_i$:

$$\ell(\theta) = \sum_{i=1}^N \left[ y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i) \right]$$

Minimizing negative log-likelihood = **minimizing binary cross-entropy**.

**Case 3: Categorical likelihood → Cross-Entropy**

For $K$-class classification:

$$\ell(\theta) = \sum_{i=1}^N \sum_{k=1}^K y_{ik} \log \hat{p}_{ik}$$

This is the **cross-entropy loss** used in virtually all classification networks.

### The Fundamental Insight

Your choice of loss function is not arbitrary — it follows directly from your assumption about the probability distribution of the data. MLE gives you the principled derivation of why each loss function takes the form it does.

---

## 11. Linear Regression — The Statistical View

### The Model

$$y = \mathbf{w}^T \mathbf{x} + b + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

We model the output $y$ as a linear function of features $\mathbf{x}$, plus Gaussian noise.

### MLE Solution

Under Gaussian noise, MLE gives us the **Ordinary Least Squares** solution:

$$\hat{\mathbf{w}} = \underset{\mathbf{w}}{\arg\min} \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2$$

The closed-form **Normal Equation** (when $N$ is small enough):

$$\hat{\mathbf{w}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

For large data, we use gradient descent instead (see Calculus section).

### The Bias-Variance Tradeoff

**Total Error = Bias² + Variance + Irreducible Noise**

$$\mathbb{E}[(\hat{f}(x) - y)^2] = \underbrace{(\mathbb{E}[\hat{f}(x)] - f(x))^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]}_{\text{Variance}} + \sigma^2$$

- **High Bias** (underfitting): Model too simple, misses patterns
- **High Variance** (overfitting): Model too complex, memorizes noise
- **Regularization** adds a bias to reduce variance

---

## 12. Logistic Regression — Probability as Output

### Why Not Linear Regression for Classification?

Linear regression outputs $(-\infty, +\infty)$. Probabilities must be in $[0, 1]$. We need a **link function**.

### The Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$$

The model becomes:

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$

### The Log-Odds (Logit)

If $p = P(y=1)$, then:

$$\text{logit}(p) = \log\frac{p}{1-p} = \mathbf{w}^T \mathbf{x} + b$$

Logistic regression models the **log-odds** as linear in features. The weights $w_j$ represent the change in log-odds per unit change in $x_j$.

### Loss Function

From MLE under Bernoulli likelihood, the **Binary Cross-Entropy Loss**:

$$\mathcal{L}(\mathbf{w}) = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

### Decision Boundary

The model predicts class 1 when $P(y=1 \mid x) > 0.5$, which is equivalent to $\mathbf{w}^T\mathbf{x} + b > 0$. This is a **hyperplane** in feature space.

Once you understand logistic regression well, more complex predictive models become natural extensions of the same ideas.

---

## 📓 Notebook

Open [`statistics_probability.ipynb`](./statistics_probability.ipynb) for hands-on code for every concept above.
