# 🔢 Linear Algebra for Machine Learning

> *"The data is the message. Linear algebra is the language."*

Linear algebra is the language of data. Data comes in tables, images, sequences — all of which are naturally represented as vectors and matrices. Understanding how to work with these objects, and what operations on them mean geometrically, is the foundation for almost all quantitative work.

---

## Table of Contents

1. [Scalars, Vectors, Matrices, Tensors](#1-scalars-vectors-matrices-tensors)
2. [Vector Operations — Geometry of Data](#2-vector-operations--geometry-of-data)
3. [Matrix Operations](#3-matrix-operations)
4. [The Dot Product & Linear Transformations](#4-the-dot-product--linear-transformations)
5. [Norms — Measuring Size](#5-norms--measuring-size)
6. [Systems of Linear Equations](#6-systems-of-linear-equations)
7. [Determinants](#7-determinants)
8. [Matrix Inverses](#8-matrix-inverses)
9. [Matrix Rank & Linear Independence](#9-matrix-rank--linear-independence)
10. [Eigenvalues & Eigenvectors](#10-eigenvalues--eigenvectors)
11. [Singular Value Decomposition (SVD)](#11-singular-value-decomposition-svd)
12. [Principal Component Analysis (PCA)](#12-principal-component-analysis-pca)

---

## 1. Scalars, Vectors, Matrices, Tensors

### The Hierarchy

| Object | Dimensions | Example |
|--------|-----------|--------|
| **Scalar** $a \in \mathbb{R}$ | 0D | A temperature reading, a price |
| **Vector** $\mathbf{v} \in \mathbb{R}^n$ | 1D | A person's measurements, a data row |
| **Matrix** $\mathbf{A} \in \mathbb{R}^{m \times n}$ | 2D | A spreadsheet, a greyscale image |
| **Tensor** $\mathcal{T} \in \mathbb{R}^{d_1 \times \ldots \times d_k}$ | $k$D | A colour image (height × width × 3) |

### Vectors as Points and Directions

A vector $\mathbf{v} = [v_1, v_2, \ldots, v_n]^T$ can mean two things:
- A **point** in $n$-dimensional space
- A **direction** with magnitude (an arrow pointing somewhere)

Context determines which interpretation makes sense.

---

## 2. Vector Operations — Geometry of Data

### Addition & Scalar Multiplication

$$\mathbf{u} + \mathbf{v} = [u_1+v_1, u_2+v_2, \ldots]^T$$
$$c\mathbf{v} = [cv_1, cv_2, \ldots]^T$$

Geometrically: addition is "tip-to-tail", scalar multiplication stretches/flips.

### Linear Combinations & Span

A **linear combination** of vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$:

$$\mathbf{u} = c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k$$

The **span** is the set of all possible linear combinations — the subspace these vectors can "reach".

---

## 3. Matrix Operations

### Transpose

$$(\mathbf{A}^T)_{ij} = A_{ji}$$

Flips rows and columns. Crucial for: weight matrices, attention, convolutions.

### Matrix Multiplication

For $\mathbf{A} \in \mathbb{R}^{m \times k}$ and $\mathbf{B} \in \mathbb{R}^{k \times n}$:

$$(\mathbf{AB})_{ij} = \sum_{l=1}^k A_{il} B_{lj}$$

The $(i,j)$ entry of $\mathbf{AB}$ is the dot product of row $i$ of $\mathbf{A}$ with column $j$ of $\mathbf{B}$.

**NOT commutative**: $\mathbf{AB} \neq \mathbf{BA}$ in general.

### Key Properties

$$(\mathbf{AB})^T = \mathbf{B}^T \mathbf{A}^T \quad \text{(order reverses)}$$
$$(\mathbf{ABC})^T = \mathbf{C}^T \mathbf{B}^T \mathbf{A}^T$$

---

## 4. The Dot Product & Linear Transformations

### Dot Product

$$\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T \mathbf{v} = \sum_{i=1}^n u_i v_i = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$$

Where $\theta$ is the angle between the vectors.

- $\mathbf{u} \cdot \mathbf{v} > 0$: vectors point in similar directions
- $\mathbf{u} \cdot \mathbf{v} = 0$: **orthogonal** (perpendicular)
- $\mathbf{u} \cdot \mathbf{v} < 0$: vectors point in opposite directions

### A Matrix as a Transformation

A matrix $\mathbf{A}$ **transforms** input vectors to output vectors. It can:
- Rotate
- Scale
- Reflect
- Project
- Any combination

This geometric view is powerful: a sequence of matrix operations is a sequence of transformations of space.

### Cosine Similarity

Normalising the dot product gives a measure of **angular similarity** between two vectors:

$$\cos(\theta) = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

This removes magnitude from the comparison — only the direction matters.

---

## 5. Norms — Measuring Size

### L2 Norm (Euclidean Length)

$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^n v_i^2} = \sqrt{\mathbf{v}^T \mathbf{v}}$$

### L1 Norm

$$\|\mathbf{v}\|_1 = \sum_{i=1}^n |v_i|$$

### General $L_p$ Norm

$$\|\mathbf{v}\|_p = \left(\sum_{i=1}^n |v_i|^p \right)^{1/p}$$

### Frobenius Norm (for Matrices)

$$\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} A_{ij}^2}$$



---

## 6. Systems of Linear Equations

### The Problem

$$\mathbf{Ax} = \mathbf{b}$$

Where $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{b} \in \mathbb{R}^m$.

Three possible outcomes:
1. **Unique solution** ($m = n$, $\mathbf{A}$ is invertible)
2. **No solution** ($\mathbf{b}$ not in the column space of $\mathbf{A}$) — overdetermined
3. **Infinitely many solutions** — underdetermined



---

## 7. Determinants

The determinant $\det(\mathbf{A})$ measures the **signed volume scaling factor** of the transformation $\mathbf{A}$.

For a $2\times2$ matrix:

$$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

### Key Properties

- $\det(\mathbf{A}) = 0$: Matrix is **singular** (collapses space, not invertible)
- $\det(\mathbf{AB}) = \det(\mathbf{A})\det(\mathbf{B})$
- $\det(\mathbf{A}^T) = \det(\mathbf{A})$
- $\det(\mathbf{A}^{-1}) = 1/\det(\mathbf{A})$



---

## 8. Matrix Inverses

For a square matrix $\mathbf{A}$:

$$\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$$

Inverse exists **iff** $\det(\mathbf{A}) \neq 0$.

### Pseudoinverse (Moore-Penrose)

When $\mathbf{A}$ is not square or not invertible:

$$\mathbf{A}^+ = \mathbf{V}\mathbf{\Sigma}^+ \mathbf{U}^T \quad \text{(from SVD)}$$

This gives the least-squares solution with minimum norm.

---

## 9. Matrix Rank & Linear Independence

### Linear Independence

Vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ are **linearly independent** if:

$$c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k = \mathbf{0} \implies c_1 = \cdots = c_k = 0$$

No vector can be written as a linear combination of the others.

### Rank

The **rank** of $\mathbf{A}$ = number of linearly independent rows = number of linearly independent columns.

$$\text{rank}(\mathbf{A}) \leq \min(m, n)$$

- **Full rank**: rank = $\min(m, n)$ — no redundant information
- **Rank deficient**: rank < $\min(m, n)$ — some dimensions carry no extra info



---

## 10. Eigenvalues & Eigenvectors

### Definition

For a square matrix $\mathbf{A}$, a nonzero vector $\mathbf{v}$ is an **eigenvector** if:

$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$

$\lambda$ is the corresponding **eigenvalue**.

Geometrically: eigenvectors are **special directions** that the transformation $\mathbf{A}$ only stretches (by $\lambda$), not rotates.

### Computing Eigenvalues

$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0 \quad \text{(characteristic equation)}$$

### Eigendecomposition

If $\mathbf{A}$ has $n$ linearly independent eigenvectors:

$$\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^{-1}$$

Where $\mathbf{Q}$ has eigenvectors as columns, $\mathbf{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_n)$.

### Symmetric Matrices (Spectral Theorem)

If $\mathbf{A} = \mathbf{A}^T$ (symmetric):
- All eigenvalues are **real**
- Eigenvectors are **orthogonal**: $\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T$

The covariance matrix $\Sigma$ is always symmetric — so its eigenvectors are guaranteed to be orthogonal and its eigenvalues real.

---

## 11. Singular Value Decomposition (SVD)

The **SVD** is the most fundamental matrix factorisation. Any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ can be decomposed as:

$$\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$$

Where:
- $\mathbf{U} \in \mathbb{R}^{m \times m}$: orthogonal matrix — **left singular vectors** (output directions)
- $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$: diagonal matrix — **singular values** $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ (scaling factors)
- $\mathbf{V}^T \in \mathbb{R}^{n \times n}$: orthogonal matrix — **right singular vectors** (input directions)

### Intuition

SVD tells you: "Here are $r$ important directions in the input space ($\mathbf{V}$), how much each is amplified ($\mathbf{\Sigma}$), and where they map to in the output space ($\mathbf{U}$)."

### Best Rank-$k$ Approximation (Eckart-Young Theorem)

The best rank-$k$ approximation of $\mathbf{A}$ in Frobenius norm:

$$\mathbf{A}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$



---

## 12. Principal Component Analysis (PCA)

PCA finds the directions (principal components) in your data that carry the most **variance**.

### Algorithm

**Step 1**: Centre the data: $\tilde{\mathbf{X}} = \mathbf{X} - \bar{\mathbf{x}}$

**Step 2**: Compute covariance matrix: $\mathbf{C} = \frac{1}{N-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}}$

**Step 3**: Eigen-decompose: $\mathbf{C} = \mathbf{Q\Lambda Q}^T$

**Step 4**: Sort by eigenvalue (descending): $\lambda_1 \geq \lambda_2 \geq \cdots$

**Step 5**: Project onto top-$k$ eigenvectors: $\mathbf{Z} = \tilde{\mathbf{X}}\mathbf{Q}_k$

### Connection to SVD

$$\tilde{\mathbf{X}} = \mathbf{U\Sigma V}^T \implies \mathbf{C} = \frac{1}{N-1}\mathbf{V\Sigma}^2\mathbf{V}^T$$

The principal components are the right singular vectors $\mathbf{V}$.

### Explained Variance

The fraction of variance explained by the first $k$ components:

$$\text{Explained Variance Ratio}_k = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^n \sigma_i^2}$$



---

## 📓 Notebook

Open [`linear_algebra.ipynb`](./linear_algebra.ipynb) for hands-on code covering all of the above.
