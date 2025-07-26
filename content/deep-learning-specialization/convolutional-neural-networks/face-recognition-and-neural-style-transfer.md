# Face Recognition and Neural Style Transfer

<!-- toc -->

## What is Face Recognition?

Face recognition is the task of identifying or verifying a person’s identity using their facial features. It can be broken down into three main categories:

- **Face Detection:** Locate faces in an image (bounding box).
- **Face Verification:** Check if two faces are of the same person (1:1 comparison).
- **Face Recognition/Identification:** Identify a person from a database (1:N comparison).

<br/>

**Real-World Applications**

- Smartphone unlock (Face ID)
- Security surveillance
- Online proctoring
- Social media tagging (e.g., Facebook)

<br/>

---

### One Shot Learning

Traditional classification algorithms require many training examples per class. However, in face recognition:

- We might only have **one image per person**.
- The task becomes: Can the model recognize a face it has seen only once?

This is known as **One-Shot Learning**.

**Problem Setup**

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/face-recognition-and-neural-style-transfer-01.png" style="display:flex; justify-content: center; width: 700px;"alt="regression-example"/>
</div>

- Instead of learning to classify, the model learns **similarity** between pairs of images.
- A **distance function** is trained to return a small value for the same person, and large for different people.

<br/>

---

### Siamese Network

A Siamese Network consists of **two identical ConvNets** (with shared weights) that compare two inputs.

<br/>

**Architecture Overview**

- Two inputs: $x_1$ and $x_2$
- Same CNN maps both to feature vectors $f(x_1)$ and $f(x_2)$
- A distance metric (e.g., L2 norm) is applied:

$$
d(x_1, x_2) = \|f(x_1) - f(x_2)\|_2^2
$$

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/face-recognition-and-neural-style-transfer-02.png" style="display:flex; justify-content: center; width: 700px;"alt="regression-example"/>
</div>

<br/>

**Loss Function**

A contrastive loss or triplet loss is used to train the network to **minimize** distances for same identities and **maximize** for different ones.

<br/>

---

### Triplet Loss

Triplet Loss is a powerful loss function for learning **embeddings**. It relies on **triplets**:

- **Anchor (A)**: A known image
- **Positive (P)**: Image of the same identity
- **Negative (N)**: Image of a different identity

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/face-recognition-and-neural-style-transfer-03.png" style="display:flex; justify-content: center; width: 500px;"alt="regression-example"/>
</div>

We want:

$$
\|f(A) - f(P)\|_2^2 + \alpha < \|f(A) - f(N)\|_2^2
$$

Where:

- $f(x)$ is the embedding function (ConvNet output)
- $\alpha$ is a margin to separate positive and negative pairs

<br/>

**Loss Function**

The Triplet Loss is:

$$
\mathcal{L}(A, P, N) = \max\left(\|f(A) - f(P)\|_2^2 - \|f(A) - f(N)\|_2^2 + \alpha, 0\right)
$$

<br/>

**Important Notes**

- **Semi-hard negative mining** improves convergence (choose negatives that are hard but not too hard).
- Embeddings are often normalized to unit length.

<br/>

---

### Face Verification and Binary Classification

Once we have embeddings from a trained network (e.g., using triplet loss), we can perform **face verification** as a binary classification task.

<br/>

**Verification Pipeline**

1. Encode both face images to embeddings.
2. Compute Euclidean distance or cosine similarity.
3. If distance < threshold $\Rightarrow$ same person.

<br/>

Threshold $\theta$ is selected based on **False Positive Rate** vs. **True Positive Rate** using ROC curve on a validation set.

<br/>

---

## What is Neural Style Transfer?

Neural Style Transfer is the task of synthesizing an image that:

- Preserves the **content** of a content image
- Adopts the **style** of a style image

Leverage a **pre-trained ConvNet** (like VGG19) to extract content and style representations.

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/face-recognition-and-neural-style-transfer-04.webp" style="display:flex; justify-content: center; width: 800px;"alt="regression-example"/>
</div>

Let:

- $C$ be the content image
- $S$ be the style image
- $G$ be the generated image

Then we optimize $G$ to minimize a cost function:

$$
J(G) = \alpha J_{content}(C, G) + \beta J_{style}(S, G)
$$

---

### What are Deep ConvNets Learning?

Deep ConvNets learn hierarchical representations:

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/face-recognition-and-neural-style-transfer-05.png" style="display:flex; justify-content: center; width: 800px;"alt="regression-example"/>
</div>

- Early layers: edges, colors, textures
- Mid layers: shapes, motifs
- Later layers: object-level concepts

In NST, **content** is encoded in deeper layers, **style** in shallower layers.

<br/>

---

### Cost Function

The **total cost** is:

$$
J(G) = \alpha J_{content}(C, G) + \beta J_{style}(S, G)
$$

Where:

- $\alpha$: weight for content preservation
- $\beta$: weight for style transfer
- Typically: $\alpha = 1$, $\beta = 10^3$ to $10^4$

<br/>

#### Content Cost Function

Let $a^{[l](C)}$ and $a^{[l](G)}$ be activations at layer $l$ for the content and generated images.

Then content cost is:

$$
J_{content}(C, G) = \frac{1}{2} \|a^{[l](C)} - a^{[l](G)}\|_2^2
$$

Use a deeper layer (e.g., `conv4_2`) for this.

<br/>

#### Style Cost Function

Style is captured by **correlations between feature maps** using a **Gram matrix**.

Let $a^{[l](S)}$ be the activations at layer $l$ for style image. Compute Gram matrix:

$$
G_{ij}^{[l]} = \sum_k a_{ik}^{[l]} a_{jk}^{[l]}
$$

Style cost is:

$$
J_{style}^{[l]}(S, G) = \frac{1}{(2n_H n_W n_C)^2} \|G^{[l](S)} - G^{[l](G)}\|_F^2
$$

Then sum over multiple layers:

$$
J_{style}(S, G) = \sum_l \lambda^{[l]} J_{style}^{[l]}(S, G)
$$

<br/>

---

## 1D and 3D Generalizations

### 1D Generalization

Neural style transfer principles can be applied to **audio** signals:

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/face-recognition-and-neural-style-transfer-06.png" style="display:flex; justify-content: center; width: 500px;"alt="regression-example"/>
</div>

- 1D convolution over waveform
- Preserve temporal content, apply style of another sound

### 3D Generalization

Applied to **volumetric data** such as:

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/face-recognition-and-neural-style-transfer-07.png" style="display:flex; justify-content: center; width: 500px;"alt="regression-example"/>
</div>

- 3D MRI scans
- 3D point clouds
- Transfer spatial styles across 3D volumes

These require 3D convolutional layers and custom Gram matrix calculations.

---

## Summary

- **Face Recognition** uses embedding learning (Triplet loss, Siamese networks).
- **One-shot learning** enables models to generalize with limited data.
- **Neural Style Transfer** uses a pre-trained CNN to blend content and style images using a combination of content/style loss.
- Both applications showcase the expressive power of deep convolutional networks beyond classic classification.
