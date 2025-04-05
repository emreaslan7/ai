# Convolutional Operations

<!-- toc -->

## Padding

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/convolution-operations-01.gif" style="display:flex; justify-content: center; width: 700px;"alt="regression-example"/>
</div>

### Why Padding is Needed

When applying convolution, the output image shrinks unless we pad it. This is a problem when building deep networks where spatial dimensions shrink after each convolution.

#### Without Padding:

$$
\text{Output size} = n - f + 1
$$

Where:

- $n$: input size
- $f$: filter size

##### Example

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/convolution-operations-03.jpeg" style="display:flex; justify-content: center; width: 500px;"alt="regression-example"/>
</div>

In this image:

- $n$: input size = $5$
- $f$: filter size = $3$

$$
\text{Output size} = n - f + 1
$$

$$
\text{Output size} = 5 - 3 + 1
$$

$$
\text{Output size} = 3
$$

#### With Padding ($p$):

$$
\text{Output size} = n + 2p - f + 1
$$

Where:

- $n$: input size
- $f$: filter size
- $p$: padding size

##### Example

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/convolution-operations-02.png" style="display:flex; justify-content: center; width: 500px;"alt="regression-example"/>
</div>

In this image:

- $n$: input size = $6$
- $f$: filter size = $3$
- $p$: padding size = $1$

$$
\text{Output size} = n + 2p - f + 1
$$

$$
\text{Output size} = 6 + (2\cdot 1) - 3 + 1
$$

$$
\text{Output size} = 6
$$

### Types of Padding

- **Valid Padding (no padding):** Output is smaller.
- **Same Padding (zero padding):** Output size equals input size.

**Real-World Analogy**

Imagine scanning a photo with a magnifying glass: without padding, you can’t examine the borders. Padding extends the image so that every pixel gets equal attention.

<br/>
<br/>
<br/>

---

<br/>

## Strided Convolutions

### What is Stride?

Stride is the number of pixels the filter moves at each step.

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/convolution-operations-04.jpeg" style="display:flex; justify-content: center; width: 700px;"alt="regression-example"/>
</div>

- **Stride = 1:** Normal convolution (moves 1 pixel at a time)
- **Stride = 2:** Downsampling (moves 2 pixels at a time)

### Output Size Formula

$$
\text{Output size} = \left\lfloor \frac{n + 2p - f}{s} \right\rfloor + 1
$$

Where:

- $n$: input size
- $f$: filter size
- $s$: stride
- $p$: padding

**Visual Example**

If stride = 2, the filter skips every alternate pixel, effectively reducing the spatial size of the output.

<br/>
<br/>
<br/>

---

<br/>

## Convolutions Over Volume

### From 2D to 3D

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/convolution-operations-05.png" style="display:flex; justify-content: center; width: 300px;"alt="regression-example"/>
</div>

In RGB images, we have 3 channels: Red, Green, and Blue. Thus, a convolutional layer operates over 3D volumes.

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/convolution-operations-07.png" style="display:flex; justify-content: center; width: 500px;"alt="regression-example"/>
</div>

### Input Dimensions:

$$
(n_H, n_W, n_C)
$$

- $n_H$: Height
- $n_W$: Width
- $n_C$: Channels (e.g., 3 for RGB)

### Filter Dimensions:

$$
(f_H, f_W, n_C)
$$

- Number of filters: $n_F$

### Output Volume:

$$
(n_H', n_W', n_F)
$$

- Each filter creates a 2D activation map, stacked together to form the output volume.

### Practical Example

Let’s say you have a (6, 6, 3) image, and you apply 2 filters of size (3, 3, 3):

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/convolution-operations-06.webp" style="display:flex; justify-content: center; width: 700px;"alt="regression-example"/>
</div>

- Output shape: (4, 4, 2) (assuming valid padding, stride=1)

<br/>
<br/>

---

<br/>
<br/>
<br/>
<br/>
