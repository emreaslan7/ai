# Anomaly Detection

<!-- toc -->

## Finding Unusual Events

Anomaly detection is the process of identifying rare or unusual patterns in data that do not conform to expected behavior. These anomalies may indicate critical situations such as fraud detection, system failures, or rare events in various fields like healthcare and finance.

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/machine-learning-specialization/anomaly-detection-02.png" style="display:flex; justify-content: center; width: 500px;"alt="regression-example"/>
</div>

### Real-World Examples

- **Credit Card Fraud Detection**: Identifying suspicious transactions that deviate significantly from a user’s normal spending habits.
- **Manufacturing Defects**: Detecting faulty products by identifying unusual patterns in production metrics.
- **Network Intrusion Detection**: Identifying cyber attacks by detecting unusual network traffic.
- **Medical Diagnosis**: Finding abnormal patterns in medical data that may indicate disease.

## Gaussian (Normal) Distribution

The Gaussian distribution, also known as the normal distribution, is a fundamental probability distribution in statistics and machine learning. It is defined as:

$$
P(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{- \frac{(x - \mu)^2}{2 \sigma^2}}
$$

Where:

- $ \mu $ is the mean (expected value)
- $ \sigma^2 $ is the variance
- $ x $ is the variable of interest

### Properties of Gaussian Distribution

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/machine-learning-specialization/anomaly-detection-01.png" style="display:flex; justify-content: center; width: 500px;"alt="regression-example"/>
</div>

- **Symmetric**: Centered around the mean $ \mu $
- **$68-95-99.7$ Rule**:
  - $68%$ of values lie within $1$ standard deviation ($ \sigma $) of the mean.
  - $95%$ within $2$ standard deviations.
  - $99.7%$ within $3$ standard deviations.

Gaussian distribution is often used in anomaly detection to model normal behavior, where deviations from this distribution indicate anomalies.

## Anomaly Detection Algorithm

### Steps in Anomaly Detection

1. **Feature Selection**: Identify relevant features from the dataset.
2. **Model Normal Behavior**: Fit a probability distribution (e.g., Gaussian) to the normal data.
3. **Calculate Probability Density**: Use the learned distribution to compute the probability density of new data points.
4. **Set a Threshold**: Define a threshold below which data points are classified as anomalies.
5. **Detect Anomalies**: Compare new observations against the threshold.

### Mathematical Approach

For a feature $ x $, assuming a Gaussian distribution:

$$

P(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{- \frac{(x - \mu)^2}{2 \sigma^2}}


$$

If $ P(x) $ is lower than a predefined threshold $ \epsilon $, then $ x $ is considered an anomaly:

$$

P(x) < \epsilon \Rightarrow x \text{ is an anomaly}


$$

## Developing and Evaluating an Anomaly Detection System

### Data Preparation

- **Obtain a labeled dataset with normal and anomalous instances**
- **Preprocess data**: Handle missing values, normalize features

### Model Training

1. Estimate parameters $ \mu $ and $ \sigma^2 $ using training data:

$$
\mu = \frac{1}{m} \sum\limits_{i=1}^{m} x^{(i)}, \quad \sigma^2 = \frac{1}{m} \sum\limits_{i=1}^{m} (x^{(i)} - \mu)^2
$$

2. Compute probability density for test data
3. Set anomaly threshold $ \epsilon $

### Performance Evaluation

- **Precision-Recall Tradeoff**: Higher recall means catching more anomalies but may include false positives.
- **F1 Score**: Harmonic mean of precision and recall.
- **ROC Curve**: Evaluates different threshold settings.

## 5. Anomaly Detection vs. Supervised Learning

| Feature                    | Anomaly Detection                      | Supervised Learning                  |
| -------------------------- | -------------------------------------- | ------------------------------------ |
| Labels Required?           | No                                     | Yes                                  |
| Works with Unlabeled Data? | Yes                                    | No                                   |
| Suitable for Rare Events?  | Yes                                    | No                                   |
| Examples                   | Fraud detection, Manufacturing defects | Spam detection, Image classification |

## Choosing What Features to Use

- **Domain Knowledge**: Understand which features are relevant.
- **Statistical Analysis**: Use correlation matrices and distributions.
- **Feature Scaling**: Normalize or standardize data.
- **Dimensionality Reduction**: Use PCA or Autoencoders to reduce noise.

## Full Python Example with TensorFlow

```python
import numpy as np
import tensorflow as tf
from scipy.stats import norm
import matplotlib.pyplot as plt

# Generate synthetic normal data
np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=1000)

# Compute mean and variance
mu = np.mean(data)
sigma = np.std(data)

# Define probability density function
pdf = norm(mu, sigma).pdf(data)

# Set anomaly threshold (e.g., 0.001 percentile)
threshold = np.percentile(pdf, 1)

# Generate new test points
new_data = np.array([30, 50, 70, 100])
new_pdf = norm(mu, sigma).pdf(new_data)

# Detect anomalies
anomalies = new_data[new_pdf < threshold]
print("Anomalies detected:", anomalies)

# Plot
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
x = np.linspace(min(data), max(data), 1000)
plt.plot(x, norm(mu, sigma).pdf(x), 'r', linewidth=2)
plt.scatter(anomalies, norm(mu, sigma).pdf(anomalies), color='red', marker='x', s=100, label='Anomalies')
plt.legend()
plt.show()
```

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/machine-learning-specialization/anomaly-detection-03.png" style="display:flex; justify-content: center; width: 500px;"alt="regression-example"/>
</div>

### Explanation

1. **Generate synthetic data**: We create a normal dataset.
2. **Compute mean and variance**: Model normal behavior.
3. **Calculate probability density**: Determine likelihood of each data point.
4. **Set threshold**: Define an anomaly cutoff.
5. **Detect anomalies**: Compare new observations against the threshold.
6. **Visualize results**: Show normal distribution and detected anomalies.

This example provides a foundation for anomaly detection using probability distributions and can be extended with deep learning techniques like autoencoders or Gaussian Mixture Models (GMMs).
