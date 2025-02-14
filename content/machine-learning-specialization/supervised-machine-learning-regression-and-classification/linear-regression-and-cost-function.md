<!-- toc -->

# Linear Regression and Cost Function

## 1. Introduction

Linear regression is one of the fundamental algorithms in machine learning. It is widely used for predictive modeling, especially when the relationship between the input and output variables is assumed to be linear. The primary goal is to find the best-fitting line that minimizes the error between predicted values and actual values.

### Why Linear Regression?

Linear regression is simple yet powerful for many real-world applications. Some common use cases include:

- **Predicting house prices** based on features like size, number of rooms, and location.
- **Estimating salaries** based on experience, education level, and industry.
- **Understanding trends** in various fields like finance, healthcare, and economics.

### Real-World Example: Housing Prices

Consider predicting house prices based on the size of the house (in square meters). A simple linear relationship can be assumed: larger houses tend to have higher prices. This assumption is the foundation of our linear regression model.

<div style="text-align: center;display:flex; justify-content: center;">
    <img src="../../../img/machine-learning-specialization/linear-regression-and-cost-function-01.png" style="display:flex; justify-content: center; width: 400px;"alt="regression-example"/>
</div>

## 2. Mathematical Representation

A simple linear regression model assumes a linear relationship between the input $x$ (house size in square meters) and the output $y$ (house price). It is represented as:

$$ h\_\theta(x) = \theta_0 + \theta_1 x $$

where:

- $h\_\theta(x) $ is the predicted house price.
- $ \theta_0 $ (intercept) and $\theta_1 $ (slope) are the parameters of the model.
- $x$ is the house size.
- $y$ is the actual house price.

### 2.1 Understanding the Linear Model

But what does this equation really mean?

- $\theta_0$ (intercept): The price of a house when its size is 0 m².

- $\theta_1$ (slope): The increase in house price for every additional square meter.

For example, if:

- $\theta_0 = 50,000$ and $\theta_1 = 300$,

- A 100 m² house would cost: $ h\_\theta(100) = 50000 + 300 \cdot 100 = 80000 $

- A 200 m² house would cost: $ h\_\theta(200) = 50000 + 300 \cdot 200 = 110000 $

We can visualize this relationship using a regression line.

## 3. Implementing Linear Regression Step by Step

To make the theoretical concepts clearer, let's implement the regression model step by step using Python.

### 3.1 Import Necessary Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### 3.2 Generate Sample Data

```python
np.random.seed(42)
x = 50 + 200 * np.random.rand(100, 1)  # House sizes in m² (50 to 250)
y = 50000 + 300 * x + np.random.randn(100, 1) * 5000  # House prices with noise
```

Here, we create a dataset with 100 samples, where:

- $x$ represents house sizes (random values between $50$ and $250$ m²).

- $y$ represents house prices, following a linear relation but with some noise.

### 3.3 Visualizing the Data

```python
plt.figure(figsize=(8,6))
sns.scatterplot(x=x.flatten(), y=y.flatten(), color='blue', alpha=0.6)
plt.xlabel('House Size (m²)')
plt.ylabel('House Price ($)')
plt.title('House Prices vs Size')
plt.show()
```

### 3.4 Plotting the Regression Line

Before moving to cost function, let's fit a simple regression line to our data and visualize it.

In real-world applications, we don't manually compute these parameters. Instead, we use libraries like **scikit-learn** to perform linear regression efficiently.

#### 3.4.1 Compute the Slope ($\theta_1$)

```python
theta_1 = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
```

Here, we compute the slope ($\theta_1$) using the **least squares method**.

#### 3.4.2 Compute the Intercept ($\theta_0$)

```python
theta_0 = np.mean(y) - theta_1 * np.mean(x)
```

This calculates the intercept ($\theta_0$), ensuring that our regression line passes through the mean of the data.

### 3.5 Plotting the Regression Line

```python
y_pred = theta_0 + theta_1 * x  # Compute predicted values

plt.figure(figsize=(8,6))
sns.scatterplot(x=x.flatten(), y=y.flatten(), color='blue', alpha=0.6, label='Actual Data')
plt.plot(x, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('House Size (m²)')
plt.ylabel('House Price ($)')
plt.title('Linear Regression Model: House Prices vs. Size')
plt.legend()
plt.show()
```

<div style="text-align: center;display:flex; justify-content: center;">
    <img src="../../../img/machine-learning-specialization/linear-regression-and-cost-function-02.png" style="display:flex; justify-content: center; width: 400px;"alt="regression-example"/>
</div>

### 3.6 Interpretation of the Regression Line

Now, what does this line tell us?

✅ If the slope $\theta_1$ is positive, then larger houses cost more (as expected).

✅ If the intercept $\theta_0$ is high, it means even the smallest houses have a significant base price.

✅ The steepness of the line shows how much price increases per square meter.

## 4. Cost Function

To measure how well our model is performing, we use the cost function. The most common cost function for linear regression is the **Mean Squared Error (MSE)**:

$$ J(\theta) = \frac{1}{2m} \sum (h\_{\theta}(x_i) - y_i)^2 $$

where:

- $ m $ is the number of training examples.
- $ h\_\theta(x_i) $ is the predicted price for the $ i-th$ house.
- $ y_i $ is the actual price.

<div style="text-align: center;display:flex; justify-content: center;">
    <img src="../../../img/machine-learning-specialization/linear-regression-and-cost-function-03.png" style="display:flex; justify-content: center; width: 400px;"alt="regression-example"/>
</div>

Any dashed line indicates an error. In the formula above, we calculated the sum of these, namely $J(\theta)$.

This function calculates the average squared difference between predicted and actual values, penalizing larger errors more. The goal is to minimize $J(\theta)$ to achieve the best model parameters.

### 4.1 Example: Assuming $\theta_1 = 0$

To illustrate how the cost function behaves, let's assume that $\theta_1 = 0$, meaning our model only depends on $\theta_0$. We'll use a small dataset with four x values and y values:

| x values | y values |
| -------- | -------- |
| 1        | 2        |
| 2        | 4        |
| 3        | 6        |
| 4        | 8        |

<div style="text-align: center;display:flex; justify-content: center; margin-top: 15px;">
    <img src="../../../img/machine-learning-specialization/linear-regression-and-cost-function-04.png" style="display:flex; justify-content: center; width: 400px;"alt="regression-example"/>
</div>

Since we assume $\theta_1 = 0$, our hypothesis function simplifies to: $$h_{\theta}(x) = \theta_0 $$

We'll evaluate different values of $\theta_0$ and compute the corresponding cost function.

#### Case 1: $\theta_0 = 1$

For $\theta_0 = 1$, the predicted values are:

$$ h\_{\theta}(x) = 1 \cdot x = [1, 2, 3, 4] $$

<div style="text-align: center;display:flex; justify-content: center; margin-top: 15px;">
    <img src="../../../img/machine-learning-specialization/linear-regression-and-cost-function-05.png" style="display:flex; justify-content: center; width: 400px;"alt="regression-example"/>
</div>

The error values:

$$ \text{error} = h\_{\theta}(x) - y = [1 - 2, 2 - 4, 3 - 6, 4 - 8] = [-1, -2, -3, -4] $$

Computing the cost function:

<div style="text-align: center;display:flex; justify-content: center; margin-top: 15px;">
    <img src="../../../img/machine-learning-specialization/linear-regression-and-cost-function-06.png" style="display:flex; justify-content: center; width: 400px;"alt="regression-example"/>
</div>

$$ J(\theta*0 = 1) = \frac{1}{2m} \sum (h*{\theta}(x_i) - y_i)^2 $$

$$ J(1) = \frac{1}{8} ((-1)^2 + (-2)^2 + (-3)^2 + (-4)^2) = \frac{1}{8} (1 + 4 + 9 + 16) = \frac{30}{8} = 3.75 $$

#### Case 2: $\theta_0 = 1.5$

For $\theta_0 = 1.5$, the predicted values are:

$$ h\_{\theta}(x) = 1.5 \cdot x = [1.5, 3, 4.5, 6] $$

<div style="text-align: center;display:flex; justify-content: center; margin-top: 15px;">
    <img src="../../../img/machine-learning-specialization/linear-regression-and-cost-function-07.png" style="display:flex; justify-content: center; width: 400px;"alt="regression-example"/>
</div>

The error values:

$$ \text{error} = [1.5 - 2, 3 - 4, 4.5 - 6, 6 - 8] = [-0.5, -1, -1.5, -2] $$

Computing the cost function:

<div style="text-align: center;display:flex; justify-content: center; margin-top: 15px;">
    <img src="../../../img/machine-learning-specialization/linear-regression-and-cost-function-08.png" style="display:flex; justify-content: center; width: 400px;"alt="regression-example"/>
</div>

$$ J(1.5) = \frac{1}{8} ((-0.5)^2 + (-1)^2 + (-1.5)^2 + (-2)^2) $$

$$ J(1.5) = \frac{1}{8} (0.25 + 1 + 2.25 + 4) = \frac{7.5}{8} = 0.9375 $$

#### Case 3: $\theta_0 = 2$ (Optimal Case)

For $\theta_0 = 2$, the predicted values match the actual values:

$$ h\_{\theta}(x) = 2 \cdot x = [2, 4, 6, 8] $$

<div style="text-align: center;display:flex; justify-content: center; margin-top: 15px;">
    <img src="../../../img/machine-learning-specialization/linear-regression-and-cost-function-09.png" style="display:flex; justify-content: center; width: 400px;"alt="regression-example"/>
</div>

The error values:

$$ \text{error} = [2 - 2, 4 - 4, 6 - 6, 8 - 8] = [0, 0, 0, 0] $$

Computing the cost function:

<div style="text-align: center;display:flex; justify-content: center; margin-top: 15px;">
    <img src="../../../img/machine-learning-specialization/linear-regression-and-cost-function-10.png" style="display:flex; justify-content: center; width: 400px;"alt="regression-example"/>
</div>

$$ J(2) = \frac{1}{8} ((0)^2 + (0)^2 + (0)^2 + (0)^2) = 0 $$

#### Comparison

From our calculations:

- $ J(1) = 3.75 $
- $ J(1.5) = 0.9375 $
- $ J(2) = 0 $

As expected, the cost function is minimized when $\theta_0 = 2$, which perfectly fits the dataset. Any deviation from this value results in a higher cost.

So how many times can the machine try and find the correct value? How can we teach it this? The answer is in the next topic.

<br/>
<br/>
