# Multiple Features

<!-- toc -->

## Introduction

In real-world scenarios, a single feature is often not enough to make accurate predictions. For example, if we want to predict the price of a house, using only its size (square meters) might not be sufficient. Other factors such as the number of bedrooms, location, and age of the house also play an important role.

When we have multiple features, our hypothesis function extends to:

$$
h_{\theta}(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n
$$

where:

- $ x_1, x_2, ..., x_n $ are the input features,
- $ \theta_0, \theta_1, ..., \theta_n $ are the parameters (weights) we need to learn.

For instance, in a house price prediction model, the hypothesis function could be:

$$
h_{\theta}(x) = \theta_0 + \theta_1 (\text{Size}) + \theta_2 (\text{Number of Bedrooms}) + \theta_3 (\text{Age of House})
$$

This allows our model to consider multiple factors, improving its accuracy compared to using a single feature.

---

## Vectorization

To optimize computations, we represent our hypothesis function using matrix notation:

where:

$ X $ is the matrix containing training examples

$ \theta $ is the parameter vector

This allows efficient computation using matrix operations instead of looping over individual training examples.

### Why Vectorization?

Vectorization is the process of converting operations that use loops into matrix operations. This improves computational efficiency, especially when working with large datasets. Instead of computing predictions one by one using a loop, we leverage linear algebra to perform all calculations simultaneously.

Without vectorization (using a loop):

```python
m = len(X)  # Number of training examples
h = []
for i in range(m):
    prediction = theta_0 + theta_1 * X[i, 1] + theta_2 * X[i, 2] + ... + theta_n * X[i, n]
    h.append(prediction)
```

With vectorization:

```python
h = np.dot(X, theta)  # Compute all predictions at once
```

This method is significantly faster because it takes advantage of optimized numerical libraries like **NumPy** that execute matrix operations efficiently.

### Vectorized Cost Function

Similarly, our cost function for multiple features is:

$$ J(\theta) = \frac{1}{2m} \sum(h_θ(x^{(i)}) - y^{(i)})^2 $$

Using matrices, this can be written as:

$$ J(\theta) = \frac{1}{2m} (X\theta - y)^T (X\theta - y) $$

And implemented in Python as:

```python
def compute_cost(X, y, theta):
    m = len(y)  # Number of training examples
    error = np.dot(X, theta) - y  # Compute (Xθ - y)
    cost = (1 / (2 * m)) * np.dot(error.T, error)  # Compute cost function
    return cost
```

By using vectorized operations, we achieve a significant performance boost compared to using explicit loops.

---

## Feature Scaling

When working with multiple features, the range of values across different features can vary significantly. This can negatively affect the performance of gradient descent, causing slow convergence or inefficient updates. **Feature scaling** is a technique used to normalize or standardize features to bring them to a similar scale, improving the efficiency of gradient descent.

### Why Feature Scaling is Important

- Features with large values can dominate the cost function, leading to inefficient updates.
- Gradient descent converges faster when features are on a similar scale.
- Helps prevent numerical instability when computing gradients.

### Methods of Feature Scaling

#### 1. **Min-Max Scaling (Normalization)**

Brings all feature values into a fixed range, typically between 0 and 1:

$$x^{(i)}_{scaled} = \frac{x^{(i)} - x_{min}}{x_{max} - x_{min}}$$

- Best for cases where the distribution of data is not Gaussian.
- Sensitive to outliers, as extreme values affect the range.

#### 2. **Standardization (Z-Score Normalization)**

Centers data around zero with unit variance:

$$x^{(i)}_{scaled} = \frac{x^{(i)} - \mu}{\sigma}$$

where:

- $ \mu $ is the mean of the feature values
- $ \sigma $ is the standard deviation

- Works well when features follow a normal distribution.
- Less sensitive to outliers compared to min-max scaling.

### Example

Consider a dataset with two features: **House Size (m²)** and **Number of Bedrooms**.

| House Size (m²) | Bedrooms |
| --------------- | -------- |
| 2100            | 3        |
| 1600            | 2        |
| 2500            | 4        |
| 1800            | 3        |

Using min-max scaling:

| House Size (scaled) | Bedrooms (scaled) |
| ------------------- | ----------------- |
| 0.714               | 0.5               |
| 0.0                 | 0.0               |
| 1.0                 | 1.0               |
| 0.286               | 0.5               |

### Feature Scaling in Gradient Descent

After scaling, gradient descent updates will be more balanced across different features, leading to faster and more stable convergence. Feature scaling is a critical preprocessing step in machine learning models involving optimization algorithms like gradient descent.

<br/>
<br/>
