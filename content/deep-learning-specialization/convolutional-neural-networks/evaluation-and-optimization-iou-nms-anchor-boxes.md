# Evaluation and Optimization: IoU, Non-max Suppression, Anchor Boxes

<!-- toc -->

## Intersection over Union (IoU)

Intersection over Union (IoU) is a metric used to evaluate the accuracy of an object detector on a particular dataset. It measures the overlap between two bounding boxes:

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/evalation-and-optimization-iou-nonmax-supperession-anchor-boxes-01.jpg" style="display:flex; justify-content: center; width: 500px;"alt="regression-example"/>
</div>

- The predicted bounding box
- The ground-truth bounding box

<br/>

**Mathematical Definition**

<br/>

If $B_p$ is the predicted bounding box and $B_{gt}$ is the ground truth bounding box:

$$
IoU = \frac{Area(B_p \cap B_{gt})}{Area(B_p \cup B_{gt})}
$$

- $IoU = 1.0$: perfect overlap
- $IoU = 0.0$: no overlap

<br/>

**Example**

Suppose:

- Predicted box: top-left = (50, 50), bottom-right = (150, 150)
- Ground truth: top-left = (100, 100), bottom-right = (200, 200)

The overlapping area is a square from (100, 100) to (150, 150) → 50x50 = 2500

Total area:

- Predicted: $100 \times 100 = 10,000$
- GT: $100 \times 100 = 10,000$
- Union: $10,000 + 10,000 - 2,500 = 17,500$

So,

$$
IoU = \frac{2500}{17500} = 0.143
$$

<br/>

**Use in Training and Evaluation**

- In training, you may ignore detections with IoU < 0.5
- For evaluation, mAP (mean average precision) uses IoU thresholds (e.g., 0.5 or 0.75)

<br/>
<br/>

---

<br/>

## Non-max Suppression (NMS)

**Why Do We Need It?**

Object detectors often output **multiple overlapping boxes** for a single object. NMS filters out redundant boxes by keeping the one with the highest confidence score.

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/evalation-and-optimization-iou-nonmax-supperession-anchor-boxes-02.webp" style="display:flex; justify-content: center; width: 700px;"alt="regression-example"/>
</div>

<br/>

**Algorithm Steps**

1. Sort all bounding boxes by their confidence score.
2. Select the box with the highest confidence and remove it from the list.
3. Compute IoU between this box and all others.
4. Remove boxes with IoU above a threshold (e.g., 0.5).
5. Repeat until no boxes remain.

<br/>

**Mathematical Intuition**

Let $B_i$ be a box with score $s_i$. You iterate over all boxes and apply:

$$
\text{Keep } B_i \text{ if } IoU(B_i, B_j) < T, \forall j < i
$$

Where $T$ is the suppression threshold.

<br/>
<br/>

---

## Anchor Boxes

**What are Anchor Boxes?**

Anchor boxes (also called prior boxes) are predefined bounding boxes with different shapes and sizes. They allow object detectors to:

- Detect **multiple objects** in the same grid cell
- Handle **aspect ratio and scale** variation

<br/>

**Why Are They Needed?**

Without anchor boxes, a single grid cell could detect only one object. But real-world scenes often contain overlapping or closely spaced objects.

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px; ">
    <img src="../../../img/deep-learning-specialization/evalation-and-optimization-iou-nonmax-supperession-anchor-boxes-03.png" style="display:flex; justify-content: center; width: 700px;"alt="regression-example"/>
</div>

<br/>

**Anchor Box Design**

You predefine $k$ anchor boxes per cell. Each one is defined by:

- Width $w$
- Height $h$
- Aspect ratio $r = \frac{w}{h}$

For example, in SSD:

- 3 feature maps
- 6 anchors per feature cell
- $\Rightarrow$ 8732 total anchor boxes

<br/>

**Output Format with Anchors**

For each anchor box, the network predicts:

- $\Delta x, \Delta y$: offset from anchor center
- $\Delta w, \Delta h$: log scale changes to width and height
- Confidence score
- Class probabilities

This transforms anchor box $(x_a, y_a, w_a, h_a)$ to the predicted box $(x_p, y_p, w_p, h_p)$:

$$
x_p = x_a + w_a \cdot \Delta x \\
y_p = y_a + h_a \cdot \Delta y \\
w_p = w_a \cdot e^{\Delta w} \\
h_p = h_a \cdot e^{\Delta h}
$$

<br/>
<br/>

---

## Summary

- **IoU** measures overlap and is used for loss/evaluation.
- **Non-max suppression** removes redundant boxes based on IoU.
- **Anchor boxes** allow detection of multiple objects at different scales/aspect ratios.

Together, these techniques form the **foundation of modern object detection pipelines** like YOLO, SSD, and Faster R-CNN.
<br/>
<br/>
