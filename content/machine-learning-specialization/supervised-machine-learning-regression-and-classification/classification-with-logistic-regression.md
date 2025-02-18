# Classification with Logistic Regression

<!-- toc -->

## 1. Introduction to Classification

Classification is a supervised learning problem where the goal is to predict **discrete categories** instead of continuous values. Unlike regression, which predicts numerical values, classification assigns data points to **labels or classes**.

### **Classification vs. Regression**

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 15px;">
    <img src="../../../img/machine-learning-specialization/classification-with-logistic-regression-01.png" style="display:flex; justify-content: center; width: 600px;"alt="regression-example"/>
</div>

| Feature           | Regression              | Classification       |
| ----------------- | ----------------------- | -------------------- |
| Output Type       | Continuous              | Discrete             |
| Example           | Predicting house prices | Email spam detection |
| Algorithm Example | Linear Regression       | Logistic Regression  |

### **Examples of Classification Problems**

- **Email Spam Detection**: Classify emails as "spam" or "not spam".
- **Medical Diagnosis**: Identify whether a patient has a disease (yes/no).
- **Credit Card Fraud Detection**: Determine if a transaction is fraudulent or legitimate.
- **Image Recognition**: Classifying images as "cat" or "dog".

Classification models can be:

- **Binary Classification**: Only two possible outcomes (e.g., spam or not spam).
- **Multi-class Classification**: More than two possible outcomes (e.g., classifying handwritten digits 0-9).

<br/>

---

## 2. Logistic Regression

### **Introduction to Logistic Regression**

Logistic regression is a statistical model used for binary classification problems. Unlike linear regression, which predicts continuous values, logistic regression predicts probabilities that map to discrete class labels.

Linear regression might seem like a reasonable approach for classification, but it has major limitations:

1. **Unbounded Output**: Linear regression produces outputs that can take any real value, meaning predictions could be **negative** or **greater than 1**, which makes no sense for probability-based classification.

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 15px;">
    <img src="../../../img/machine-learning-specialization/classification-with-logistic-regression-03.jpeg" style="display:flex; justify-content: center; width: 600px;"alt="regression-example"/>
</div>

2. **Poor Decision Boundaries**: If we use a linear function for classification, extreme values in the dataset can distort the decision boundary, leading to incorrect classifications.

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 15px;">
    <img src="../../../img/machine-learning-specialization/classification-with-logistic-regression-04.png" style="display:flex; justify-content: center; width: 400px;"alt="regression-example"/>
    <img src="../../../img/machine-learning-specialization/classification-with-logistic-regression-05.png" style="display:flex; justify-content: center; width: 400px;"alt="regression-example"/>
</div>

To solve these issues, we use **logistic regression**, which applies the **sigmoid function** to transform outputs into a probability range between **0 and 1**.

---

### **Why Do We Need the Sigmoid Function?**

The **sigmoid function** is a key component of logistic regression. It ensures that outputs always remain between **0 and 1**, making them interpretable as probabilities.

Consider a **fraud detection system** that predicts whether a transaction is fraudulent (1) or legitimate (0) based on customer behavior. Suppose we use a linear model:

$$
y = \theta_0 + \theta_1 x_1 + \theta_2 x_2
$$

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px;">
    <img src="../../../img/machine-learning-specialization/classification-with-logistic-regression-02.png" style="display:flex; justify-content: center; width: 600px;"alt="regression-example"/>
</div>

For some transactions, the output might be **y = 7.5** or **y = -3.2**, which do not make sense as probability values. Instead, we use the **sigmoid function** to squash any real number into a valid probability range:

$$
h_{\theta}(x) = \frac{1}{1 + e^{-\theta^T x}}
$$

This function maps:

- Large positive values to probabilities close to **1** (fraudulent transaction).
- Large negative values to probabilities close to **0** (legitimate transaction).
- Values near **0** to probabilities near **0.5** (uncertain classification).

---

### **Sigmoid Function and Probability Interpretation**

The output of the sigmoid function can be interpreted as:

- **$ h_θ(x) \approx 1 $** → The model predicts **Class 1** (e.g., spam email, fraudulent transaction).
- **$ h_θ(x) \approx 0 $** → The model predicts **Class 0** (e.g., not spam email, legitimate transaction).

For a final classification decision, we apply a **threshold** (typically 0.5):

$$
\hat{y} =
\begin{cases}
1, & \text{if } h_{\theta}(x) \geq 0.5 \\
0, & \text{if } h_{\theta}(x) < 0.5
\end{cases}
$$

This means:

- If the probability is **≥ 0.5**, we classify the input as **1 (positive class)**.
- If the probability is **< 0.5**, we classify it as **0 (negative class)**.

---

### **Decision Boundary**

The **decision boundary** is the surface that separates different classes in logistic regression. It is the point at which the model predicts a probability of **0.5**, meaning the model is equally uncertain about the classification.

Since logistic regression produces probabilities using the **sigmoid function**, we define the decision boundary mathematically as:

$$
h_{\theta}(x) = \frac{1}{1 + e^{-\theta^T x}} = 0.5
$$

Taking the inverse of the sigmoid function, we get:

$$
\theta^T x = 0
$$

This equation defines the decision boundary as a **linear function** in the feature space.

---

#### **Understanding the Decision Boundary with Examples**

##### **1. Single Feature Case (1D)**

If we have only **one feature** $ x_1 $, the model equation is:

$$
\theta_0 + \theta_1 x_1 = 0
$$

Solving for $ x_1 $:

$$
x_1 = -\frac{\theta_0}{\theta_1}
$$

This means that when $ x_1 $ crosses this threshold, the model switches from predicting **Class 0** to **Class 1**.

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px;">
    <img src="../../../img/machine-learning-specialization/classification-with-logistic-regression-06.png" style="display:flex; justify-content: center; width: 400px;"alt="regression-example"/>
</div>

**Example:**
Imagine predicting whether a student passes or fails based on study hours ($ x_1 $):

- If $ x_1 < 5 $ hours → Fail (Class 0).
- If $ x_1 \geq 5 $ hours → Pass (Class 1).

The decision boundary in this case is simply $ x_1 = 5 $.

---

##### **2. Two Features Case (2D)**

For **two features** $ x_1 $ and $ x_2 $, the decision boundary equation becomes:

$$
\theta_0 + \theta_1 x_1 + \theta_2 x_2 = 0
$$

Rearranging:

$$
x_2 = -\frac{\theta_0}{\theta_2} - \frac{\theta_1}{\theta_2} x_1
$$

This represents a **straight line** separating the two classes in a **2D plane**.

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px;">
    <img src="../../../img/machine-learning-specialization/classification-with-logistic-regression-07.png" style="display:flex; justify-content: center; width: 400px;"alt="regression-example"/>
</div>

**Example:**
Suppose we classify students as passing (1) or failing (0) based on **study hours ($ x_1 $)** and **sleep hours ($ x_2 $)**:

- The decision boundary could be:
  $$
  x_2 = -2 - 0.5 x_1
  $$
- If $ x_2 $ is above the line, classify as **pass**.
- If $ x_2 $ is below the line, classify as **fail**.

---

##### **3. Two Features Case (3D)**

When we move to **three features** $ x_1 $, $ x_2 $, and $ x_3 $, the decision boundary becomes a **plane** in three-dimensional space:

$$
\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 = 0
$$

Rearranging for $ x_3 $:

$$
x_3 = -\frac{\theta_0}{\theta_3} - \frac{\theta_1}{\theta_3} x_1 - \frac{\theta_2}{\theta_3} x_2
$$

This equation represents a **flat plane** dividing the 3D space into two regions, one for **Class 1** and the other for **Class 0**.

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px;">
    <img src="../../../img/machine-learning-specialization/classification-with-logistic-regression-08.png" style="display:flex; justify-content: center; width: 400px;"alt="regression-example"/>
</div>

**Example:**  
Imagine predicting whether a company will be **profitable (1) or not (0)** based on:

- **Marketing Budget** ($ x_1 $)
- **R&D Investment** ($ x_2 $)
- **Number of Employees** ($ x_3 $)

The decision boundary would be a **plane** in 3D space, separating profitable and non-profitable companies.

In general, for **n features**, the decision boundary is a **hyperplane** in an n-dimensional space.

---

##### **4. Non-Linear Decision Boundaries in Depth**

So far, we have seen that **logistic regression** creates **linear** decision boundaries. However, many real-world problems have **non-linear** relationships. In such cases, a straight line (or plane) is **not sufficient** to separate classes.

To capture **complex decision boundaries**, we introduce **polynomial features** or **feature transformations**.

###### **Example 1: Circular Decision Boundary**

If the data requires a **circular boundary**, we can use quadratic terms:

$$
\theta_0 + \theta_1 x_1^2 + \theta_2 x_2^2 = 0
$$

This represents a **circle** in 2D space.

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px;">
    <img src="../../../img/machine-learning-specialization/classification-with-logistic-regression-10.png" style="display:flex; justify-content: center; width: 400px;"alt="regression-example"/>
</div>

For example:

- If $ x_1 $ and $ x_2 $ are the coordinates of points, a decision boundary like:

  $$
  x_1^2 + x_2^2 = 4
  $$

  would classify points inside a **radius-2 circle** as Class 1 and outside as Class 0.

###### **Example 2: Elliptical Decision Boundary**

A more general quadratic equation:

$$
\theta_0 + \theta_1 x_1^2 + \theta_2 x_2^2 + \theta_3 x_1 x_2 = 0
$$

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px;">
    <img src="../../../img/machine-learning-specialization/classification-with-logistic-regression-11.png" style="display:flex; justify-content: center; width: 400px;"alt="regression-example"/>
</div>

This allows for **elliptical** decision boundaries.

###### **Example 3: Complex Non-Linear Boundaries**

For even more **complex** boundaries, we can include **higher-order polynomial features**, such as:

$$
\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_1^2 + \theta_4 x_2^2 + \theta_5 x_1 x_2 + \theta_6 x_1^3 + \theta_7 x_2^3 = 0
$$

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px;">
    <img src="../../../img/machine-learning-specialization/classification-with-logistic-regression-09.png" style="display:flex; justify-content: center; width: 400px;"alt="regression-example"/>
</div>

This enables **twists and curves** in the decision boundary, allowing logistic regression to model **highly non-linear** patterns.

###### **Feature Engineering for Non-Linear Boundaries**

- Instead of adding polynomial terms manually, we can **transform** features using **basis functions** (e.g., Gaussian kernels or radial basis functions).
- **Feature maps** can convert non-linearly separable data into a higher-dimensional space where a linear decision boundary works.

###### **Limitations of Logistic Regression for Non-Linear Boundaries**

- **Feature engineering is required**: Unlike neural networks or decision trees, logistic regression cannot learn complex boundaries automatically.
- **Higher-degree polynomials can lead to overfitting**: Too many non-linear terms make the model sensitive to noise.

---

### **Key Takeaways**

- In **3D**, the decision boundary is a **plane**, and in higher dimensions, it becomes a **hyperplane**.
- **Non-linear decision boundaries** can be created using **quadratic, cubic, or transformed features**.
- **Feature engineering is crucial** to make logistic regression work well for non-linearly separable problems.
- **Too many high-order polynomial terms** can cause overfitting, so regularization is needed.

<br/>
<br/>

---

## 3. Cost Function for Logistic Regression

### **1. Why Do We Need a Cost Function?**

In linear regression, we use the **Mean Squared Error (MSE)** as the cost function:

$$
J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_θ(x_i) - y_i)^2
$$

However, this cost function does not work well for **logistic regression** because:

- The hypothesis function in logistic regression is **non-linear** due to the sigmoid function.
- Using squared errors results in a **non-convex** function with multiple local minima, making optimization difficult.

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px;">
    <img src="../../../img/machine-learning-specialization/classification-with-logistic-regression-12.png" style="display:flex; justify-content: center; width: 400px;"alt="regression-example"/>
</div>

We need a different cost function that:  
✅ Works well with the **sigmoid function**.  
✅ Is **convex**, so gradient descent can efficiently minimize it.

---

### **2. Simplified Cost Function for Logistic Regression**

Instead of using squared errors, we use a **log loss function**:

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(h_θ(x_i)) + (1 - y_i) \log(1 - h_θ(x_i)) \right]
$$

Where:

- $ y_i $ is the true label (0 or 1).
- $ h_θ(x_i) $ is the predicted probability from the sigmoid function.

This function ensures:

- **If $ y = 1 $** → The first term dominates: $ -\log(h_θ(x)) $, which is close to 0 if $ h\_\theta(x) \approx 1 $ (correct prediction).
- **If $ y = 0 $** → The second term dominates: $ -\log(1 - h_θ(x)) $, which is close to 0 if $ h\_\theta(x) \approx 0 $.

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/machine-learning-specialization/classification-with-logistic-regression-13.png" style="display:flex; justify-content: center; width: 700px;"alt="regression-example"/>
</div>

✅ **Interpretation**: The function penalizes incorrect predictions heavily while rewarding correct predictions.

---

### **3. Intuition Behind the Cost Function**

Let’s break it down:

- When **$ y = 1 $**, the cost function simplifies to:

  $$
  -\log(h_θ(x))
  $$

  This means:

  - If $ h_θ(x) \approx 1 $ (correct prediction), $ -\log(1) = 0 $ → No penalty.
  - If $ h_θ(x) \approx 0 $ (wrong prediction), $ -\log(0) \to \infty $ → High penalty!

- When **$ y = 0 $**, the cost function simplifies to:

  $$
  -\log(1 - h_θ(x))
  $$

  This means:

  - If $ h_θ(x) \approx 0 $ (correct prediction), $ -\log(1) = 0 $ → No penalty.
  - If $ h_θ(x) \approx 1 $ (wrong prediction), $ -\log(0) \to \infty $ → High penalty!

✅ **Key Takeaway**:  
The function assigns very high penalties for incorrect predictions, encouraging the model to learn correct classifications.

<br/>
<br/>

---

## 4. Gradient Descent for Logistic Regression

### **1. Why Do We Need Gradient Descent?**

In logistic regression, our goal is to find the best **parameters** $ \theta $ that minimize the **cost function**:

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(h_{\theta}(x_i)) + (1 - y_i) \log(1 - h_{\theta}(x_i)) \right]
$$

Since there is no **closed-form solution** like in linear regression, we use **gradient descent** to iteratively update $ \theta $ until we reach the minimum cost.

---

### **2. Gradient Descent Algorithm**

Gradient descent updates the parameters using the rule:

$$
\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
$$

Where:

- $ \alpha $ is the **learning rate** (step size).
- $ \frac{\partial J(\theta)}{\partial \theta_j} $ is the **gradient** (direction of steepest increase).

For logistic regression, the derivative of the cost function is:

$$
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x_i) - y_i) x_{ij}
$$

Thus, the update rule becomes:

$$
\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x_i) - y_i) x_{ij}
$$

✅ **Key Insight:**

- We compute the error: $ h_θ(x_i) - y_i $.
- Multiply it by the feature $ x\_{ij} $.
- Average over all training examples.
- Scale by $ \alpha $ and update $ \theta_j $.

---
