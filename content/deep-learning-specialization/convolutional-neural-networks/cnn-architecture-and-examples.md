# CNN Architecture and Examples

<!-- toc -->

## 1. One Layer of a Convolutional Network

A **Convolutional Neural Network (CNN)** is typically composed of three types of layers:

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/cnn-architecture-and-examples-01.webp" style="display:flex; justify-content: center; width: 700px;"alt="regression-example"/>
</div>

- **Convolutional layers:** Apply filters to extract spatial features.
- **Pooling layers:** Downsample feature maps to reduce computation.
- **Fully connected layers:** Perform final classification or regression.

Each layer transforms the input volume into an output volume through learnable parameters or fixed operations.

### Layer Types of CNN

#### 1. Convolutional Layers

**Purpose:**

To extract spatial features such as edges, textures, and patterns by sliding filters over the input image or feature map.

**How it works:**

- A filter (or kernel) of size $f \times f$ slides over the input.
- At each location, an element-wise multiplication is performed between the filter and the part of the input it overlaps.
- The results are summed to produce a single number in the output feature map.

**Mathematical Operation:**

Let the input be $X \in \mathbb{R}^{n_H \times n_W \times n_C}$ and the filter be $W \in \mathbb{R}^{f \times f \times n_C}$.

$$
Z_{i,j} = \sum_{m=0}^{f-1} \sum_{n=0}^{f-1} \sum_{c=0}^{n_C-1} X_{i+m,j+n,c} \cdot W_{m,n,c} + b
$$

**Example:**

Input: $5 \times 5$ grayscale image with $3 \times 3$ filter:

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/cnn-architecture-and-examples-02.gif" style="display:flex; justify-content: center; width: 300px;"alt=""/>
</div>

As the filter slides across the input, it detects vertical and horizontal edges by producing high activation in regions with strong center transitions.

---

#### 2. Pooling Layers

**Purpose:**

To reduce the spatial dimensions (height and width) of the feature maps, thereby:

- Reducing the number of parameters and computation
- Controlling overfitting
- Making the model invariant to small translations in the input

**Types:**

**Max Pooling:**

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/cnn-architecture-and-examples-03.gif" style="display:flex; justify-content: center; width: 300px;"alt=""/>
</div>

Selects the maximum value in each region.

**Average Pooling:**

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/cnn-architecture-and-examples-04.gif" style="display:flex; justify-content: center; width: 300px;"alt=""/>
</div>

Takes the average of values in each region.

---

#### 3. Fully Connected Layers

To connect every neuron in one layer to every neuron in the next layer, performing the final classification or regression.

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/cnn-architecture-and-examples-05.png" style="display:flex; justify-content: center; width: 500px;"alt=""/>
</div>

**How it works:**

- Takes the flattened output from the last convolutional/pooling layer
- Passes it through one or more dense layers
- Final layer often uses **softmax** for classification

**Mathematical Form:**

Given the input vector $x \in \mathbb{R}^n$, weights $W \in \mathbb{R}^{m \times n}$, and bias $b \in \mathbb{R}^m$:

$$
z = Wx + b
$$

$$
a = g(z) \text{ where } g \text{ is an activation function (e.g., ReLU, Softmax)}
$$

**Example:**

Let’s say we have a feature map output size of $5 \times 5 \times 16 = 400$ from the last pooling layer:

- FC1: 400 → 120 (ReLU)
- FC2: 120 → 84 (ReLU)
- FC3: 84 → 10 (Softmax, for 10-class classification)

These dense layers combine all the high-level features learned in the earlier layers and output a prediction.

---

#### Summary Table

| Layer Type      | Role                            | Typical Parameters     | Output Shape Transformation                                                |
| --------------- | ------------------------------- | ---------------------- | -------------------------------------------------------------------------- |
| Convolutional   | Extract local spatial features  | $f$, $s$, $p$, filters | $n_H \times n_W \times n_C \rightarrow n_{H'} \times n_{W'} \times n_{C'}$ |
| Pooling         | Downsample feature maps         | $f$, $s$               | $n_H \times n_W \times n_C \rightarrow n_{H'} \times n_{W'} \times n_C$    |
| Fully Connected | Final classification/regression | neurons per layer      | $n \rightarrow m$ (vector size)                                            |

These layers together form the foundation of Convolutional Neural Networks, enabling them to learn hierarchical representations from raw pixels to abstract concepts.

### Notation and Terminology

- $ n_H, n_W $: height and width of the input volume
- $ n_C $: number of channels (depth)
- $ f $: filter size
- $ s $: stride
- $ p $: padding
- $ W^{[l]} $, $ b^{[l]} $: weights and biases at layer $ l $

### Parameters and Learnable Components

- **Weights ($ W $)**: Represent filters; shared spatially across the input.
- **Biases ($ b $)**: One per filter.
- **Activation ($ A $)**: Output of ReLU or other non-linear function.

Each neuron in a layer is connected only to a small region of the previous layer, leading to sparse interactions and parameter sharing.

---

## 3. CNN Example (Comprehensive Network)

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/cnn-architecture-and-examples-06.webp" style="display:flex; justify-content: center; width: 750px;"alt=""/>
</div>

<br/>
<br/>

## Why Convolutions?

Convolutional layers are the **cornerstone of modern deep learning models in computer vision**, replacing traditional fully connected layers in image tasks. This section explores **why** convolutions are used instead of dense layers, and what advantages they bring.

---

### 1. The Limitations of Fully Connected Layers for Images

**a. Parameter Explosion**

A fully connected (dense) layer connecting every pixel of an image to every neuron in the next layer requires a **huge number of parameters**.

Example:

- Input image size: $ 64 \times 64 \times 3 = 12,288 $
- Fully connected layer with 1000 neurons:
  $ \text{Parameters} = 12,288 \times 1000 = 12,288,000 $

This leads to high memory usage, overfitting risk, and long training times.

**b. Ignores Spatial Structure**

Dense layers treat input features independently and do not take advantage of the **spatial locality** of image data.

- A cat’s ear in the top-left and bottom-right corners are treated as unrelated by dense layers.

---

### 2. Benefits of Convolutional Layers

**a. Sparse Interactions**

Each output neuron is connected only to a **small region** of the input (called the **receptive field**).

- Fewer parameters
- Faster computations

Example:

- Using $ f = 5 $ instead of connecting all 12,288 pixels

**b. Parameter Sharing**

Same filter (weights) is applied across the entire image:

$$
Z[i, j] = \sum_{m=0}^{f-1} \sum_{n=0}^{f-1} W[m, n] \cdot X[i+m, j+n] + b
$$

This results in **drastic reduction** in number of parameters and allows **feature detection to be translation invariant**.

**c. Translation Equivariance**

- If an object moves in the image, its feature map also moves.
- The model learns **position-independent** features — important for generalization.

<br/>
<br/>
<br/>

---

<br/>
<br/>
<br/>
<br/>
