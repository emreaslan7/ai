# Region Proposals and Semantic Segmentation: U-Net

<!-- toc -->

## Region Proposals

### Why Region Proposals?

Traditional object detectors like sliding windows are computationally expensive due to scanning every possible region in the image. **Region Proposal methods** address this by generating a small number of candidate regions likely to contain objects.

### Selective Search

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/region-proposal-and-semantic-segmentation-unet.png" style="display:flex; justify-content: center; width: 800px;"alt="regression-example"/>
</div>

- Group similar pixels into **superpixels**
- Merge regions based on similarity
- Outputs ~2000 proposals per image

### R-CNN Pipeline

1. Use **Selective Search** to propose regions.
2. Warp each region to a fixed size (e.g., 224x224).
3. Pass through a ConvNet to extract features.
4. Use SVMs for classification and regressors for bounding boxes.

> Limitation: Very slow due to independent ConvNet run on each region.

---

## Semantic Segmentation

### What is Semantic Segmentation?

Semantic segmentation is the task of classifying each pixel of an image into a class label.

- **Image Classification**: What is in the image?
- **Object Detection**: Where is the object?
- **Semantic Segmentation**: Which pixel belongs to which class?

### Applications

- Medical imaging (e.g., tumor segmentation)
- Autonomous driving (lane and pedestrian detection)
- Satellite image analysis
- Industrial defect detection

---

## Transpose Convolutions (Deconvolution)

### Motivation

In segmentation tasks, we need to **upsample** feature maps back to the original image size. Transpose convolutions (a.k.a. deconvolutions) help with this.

### How It Works

A transpose convolution is the reverse of a normal convolution:

- While convolution **reduces** spatial size (downsampling),
- Transpose convolution **increases** it (upsampling).

### Mathematical Operation

Suppose an input size of $N \times N$ and a kernel size of $k \times k$ with stride $s$.

- Convolution output size:

  $$
  O = \left\lfloor \frac{N - k}{s} + 1 \right\rfloor
  $$

- Transpose convolution (reverses the above):
  $$
  O_{up} = (N - 1) \cdot s + k
  $$

### Alternatives

- Nearest-neighbor or bilinear upsampling + 1x1 conv (cheaper, less expressive)
- Learned transpose convolutions (richer)

---

## U-Net Architecture Intuition

**Key Idea**

U-Net is a fully convolutional network that consists of:

- A **contracting path** to capture context (downsampling)
- An **expanding path** to enable precise localization (upsampling)

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/region-proposal-and-semantic-segmentation-unet-02.png" style="display:flex; justify-content: center; width: 800px;"alt="regression-example"/>
</div>

U-Net was originally designed for **biomedical image segmentation** but is now used in many fields.

### Contracting Path (Encoder)

- Similar to standard CNN (e.g., VGG)
- Repeated 2x:
  - Conv (ReLU) → Conv (ReLU) → MaxPooling

### Expanding Path (Decoder)

- Transpose convolution for upsampling
- Skip connections concatenate features from encoder

### Why Skip Connections?

Skip connections pass high-resolution features from encoder to decoder, enabling:

- Better boundary localization
- Preservation of fine details

---

## U-Net Architecture (Full Design)

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/region-proposal-and-semantic-segmentation-unet-03.png" style="display:flex; justify-content: center; width: 800px;"alt="regression-example"/>
</div>

### Structure Overview

- Input size: $572 \times 572$
- Each layer: two $3 \times 3$ convolutions + ReLU
- Downsampling: $2 \times 2$ max-pooling
- Upsampling: transpose convolutions
- Final output: $1 \times 1$ convolution to map to $C$ classes (per pixel)

### Example Architecture

```plaintext
Input → Conv → Conv → Pool
      ↓             ↑
     Conv → Conv → Pool
      ↓             ↑
     Conv → Conv → Pool
      ↓             ↑
     Bottleneck     ← Skip Connections
      ↓             ↑
     Upconv → Concat → Conv → Conv
      ↓
    Output (Segmentation Map)
```

### Loss Function

Typical loss: **Pixel-wise cross-entropy loss**.

$$
\mathcal{L} = - \sum_{i=1}^{H} \sum_{j=1}^{W} \sum_{c=1}^{C} y_{ij}^{(c)} \log(\hat{y}_{ij}^{(c)})
$$

Where:

- $H, W$: height and width of the image
- $C$: number of classes
- $y_{ij}^{(c)}$: ground truth indicator (1 if pixel $(i,j)$ belongs to class $c$)
- $\hat{y}_{ij}^{(c)}$: predicted probability for class $c$ at pixel $(i,j)$

### Performance Metrics

- **Pixel Accuracy**: overall correct classification
- **IoU per class**: same as object detection, applied per-pixel
- **Dice Coefficient**: common in medical segmentation

---

## Summary

- Region proposals are key to efficient object detection pipelines like R-CNN.
- Semantic segmentation classifies each pixel and requires upsampling layers.
- Transpose convolutions allow learned upsampling.
- U-Net combines low-level and high-level features through skip connections and is state-of-the-art for many segmentation tasks.
