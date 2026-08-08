# SIFT Detector and Descriptor

<!-- toc -->

## 1. Overview

In traditional computer vision approaches, **binary segmentation** and **geometric moment analysis** are quite effective for object recognition and localization. However, these methods only demonstrate stability in strictly controlled industrial environments (backlit silhouettes) or high-contrast text extraction applications (license plate recognition, etc.).

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-01.png" alt="Simple Template vs Complex 2D Appearance Matching" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 1: (Left) Isolated single template cover. (Right) Complex real-world 2D scene containing overlapping and rotated CD covers.</em></figcaption>
  </div>
</figure>

When it comes to recognizing three-dimensional or complex planar two-dimensional objects in real-world scenes, these simplistic approaches fail completely.

```
Limitations of Traditional Template Matching:
─────────────────────────────────────────────
1. Scale Changes: Variations in object size due to depth.
2. Rotation: 2D in-plane and 3D out-of-plane rotations.
3. Occlusion: Partial blockage of the object of interest.
4. Illumination: Variations in lighting, specularities, and camera gain.
```

If one attempts to use classic **template matching** or **normalized cross-correlation (NCC)** to find an object, thousands of partial sub-templates must be generated for all possible rotation angles and scale factors, and slid across the entire image. This process reaches a computational complexity of $O(N \cdot M \cdot S \cdot R)$, making it completely intractable for practical systems.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-02.png" alt="Appearance under Rotation and Illumination Changes" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 2: Upright object orientation (left) versus rotated and re-illuminated orientation (right). Direct local patch pixel values cannot be matched.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-03.png" alt="Comparison of Zoomed-in Pixel Patches" style="display:flex; border-radius: 5px; justify-content: center; width: 450px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 3: Zoomed-in local pixel patch. When an object rotates, the spatial arrangement of the pixel matrix changes completely, causing pixel-wise differencing to fail.</em></figcaption>
  </div>
</figure>

> **Key Insight:** Overcoming this fundamental problem relies on extracting **highly descriptive and unique local features** directly from the image that are invariant to geometric and photometric transformations. Once their spatial coordinates and local appearance signatures (descriptors) are extracted, keypoints across different images can be matched one-to-one for object recognition, image stitching, and 3D reconstruction.

---

## 2. What is an Interest Point?

An **interest point** in an image is a local region that possesses the richest visual information and uniqueness. For a local patch to qualify as an interest point, it must fulfill several critical criteria:

### Desirable Properties of an Ideal Interest Point:
- **Rich Content:** The local analysis window must contain high variance in intensity/color.
- **Well-defined Representation:** A compact, distinctive visual signature (descriptor) must be computable from the local texture around the point for matching.
- **Well-defined Position:** The interest point must have a precise spatial coordinate ($x, y$) in the image plane for spatial accuracy.
- **Scale and Rotation Invariance:** Even when the object scales up/down or rotates, the same spatial location and signature must be reliably detected (repeatability).
- **Insensitivity to Illumination:** It must remain stable under shadows, specular highlights, and camera gain adjustments.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-04.png" alt="Homogeneous and Flat Texture Patches" style="display:flex; border-radius: 5px; justify-content: center; width: 450px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 4: Flat and homogeneous texture patches (wood grain / flat surface). Lacking gradient variance, they cannot serve as interest points.</em></figcaption>
  </div>
</figure>

### Evaluating Lines, Edges, Corners, and Blobs:

1. **Edges:** Edges are regions where intensity changes rapidly in a single direction. Sliding a local window along an edge line reveals virtually no appearance change (**aperture problem**). This spatial ambiguity makes edges unsuitable as interest points.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-05.png" alt="Edge Detection and the Aperture Problem" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 5: Sliding ambiguity along straight edges (Aperture Problem). Moving the window along the edge line leaves local pixel values unchanged, preventing precise spatial localization.</em></figcaption>
  </div>
</figure>

2. **Corners:** Corners represent the intersection of two distinct edge directions, providing well-defined spatial localization ($x, y$). However, they lack sufficiently rich local appearance information to represent complex textured objects and occur sparsely.
3. **Blobs and Patches:** Circular or elliptical patches characterized by a specific spatial scale ($\sigma$), a dominant orientation ($\theta$), and rich internal texture variation. Because their location, scale, and local texture can be mathematically modeled with high stability, **Blob** structures represent the ideal interest point candidate in computer vision.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-06.png" alt="Corner and Blob Patch Analysis" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 6: Comparison of corner and blob patches against flat regions. Blob patches provide both well-defined spatial localization and a well-scaled appearance window.</em></figcaption>
  </div>
</figure>

---

## 3. Detecting Blobs

Mathematically, detecting a blob corresponds to finding local intensity extrema (peaks) across different spatial resolution levels (**scale-space**).

### 3.1 1D Signal Second Derivatives and Scale-Space

In a 1D signal, noise is smoothed using a Gaussian filter of standard deviation $\sigma$:

$$G(x, \sigma) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{2\sigma^2}}$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-07.png" alt="1D Signal Gaussian Smoothing" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 7: (Top to bottom) Noisy step edge signal $f$, Gaussian kernel $n_\sigma$, and smoothed signal $n_\sigma * f$.</em></figcaption>
  </div>
</figure>

Convolving the signal with the first derivative of a Gaussian ($\frac{d}{dx} G_\sigma$) produces a peak response at step transitions.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-08.png" alt="Gaussian First Derivative Response" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 8: 1st derivative of Gaussian $\nabla(n_\sigma)$ filter response, forming a peak amplitude precisely over the edge.</em></figcaption>
  </div>
</figure>

Applying the second derivative of a Gaussian ($\frac{d^2}{dx^2} G_\sigma$ / Inverted Mexican Hat) yields a **Zero-Crossing** at the exact center of the edge.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-09.png" alt="Gaussian Second Derivative Zero-Crossing" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 9: 2nd derivative of Gaussian $\nabla^2(n_\sigma)$ filter and its convolution result, demonstrating a zero-crossing centered over the edge transition.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-10.png" alt="Examples of 1D Blob Structures" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 10: Typical 1D blob-like signal structures (pulses, troughs, bumps).</em></figcaption>
  </div>
</figure>

To analyze blobs of varying widths (e.g., Blobs $A, B, C$ with widths $W, 2W, 3W$), a **Scale-Space** is constructed by continuously increasing the filter standard deviation ($\sigma$):

$$S(x, \sigma) = f(x) * G(x, \sigma)$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-11.png" alt="Filter Responses on Blobs of Different Widths" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 11: Blobs of different widths ($A, B, C$) evaluated under Gaussian smoothing and second derivatives. Without normalization, response amplitudes decay at higher scales.</em></figcaption>
  </div>
</figure>

### 3.2 $\sigma^2$-Normalization and Characteristic Scale

As the Gaussian standard deviation ($\sigma$) increases (coarser scale), the peak amplitude of the filter decreases, dampening the response. To compare extrema across different scale levels consistently, the second derivative filter is multiplied by a scaling factor of $\sigma^2$. This yields the **$\sigma$-normalized derivative response**:

$$\text{NLoG}_{1D} = \sigma^2 \frac{d^2 G_\sigma}{dx^2} * f(x)$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-12.png" alt="Characteristic Scale and Local Extrema" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 12: $\sigma^2$-normalized NLoG response forming a maximum extremum at the exact spatial center of each blob.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-13.png" alt="Relationship between Blob Size and Characteristic Scale" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 13: Characteristic Scale ($\sigma^*$): Maximum response is achieved at $\sigma_1$ for Blob $A$, $2\sigma_1$ for Blob $B$, and $3\sigma_1$ for Blob $C$.</em></figcaption>
  </div>
</figure>

Plotting the normalized response amplitude at a blob's center across values of $\sigma$ reveals a maximum (local extremum) at a scale proportional to the blob's spatial size ($\sigma^* \propto \text{Blob Width}$):

- **Blob $A$ ($\text{Width}=W$):** Peak response at $\sigma_A^* = \sigma_1$.
- **Blob $B$ ($\text{Width}=2W$):** Peak response at $\sigma_B^* = 2\sigma_1$.
- **Blob $C$ ($\text{Width}=3W$):** Peak response at $\sigma_C^* = 3\sigma_1$.

> **Characteristic Scale:** The unique scale $\sigma^*$ where the normalized operator reaches a local maximum is called the Characteristic Scale. Searching for extrema in 2D $(x, \sigma)$-space simultaneously yields both the exact spatial location ($x^*$) and the true physical scale ($\sigma^*$) of the blob.

### 3.3 2D Normalized Laplacian of Gaussian (NLoG)

In 2D images, the equivalent of the 1D normalized second derivative is the **Normalized Laplacian of Gaussian (NLoG)** operator. It is formed by taking the Laplacian ($\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$) of a 2D Gaussian and scaling by $\sigma^2$:

$$\text{NLoG}_{2D} = \sigma^2 \nabla^2 G(x, y, \sigma) = \sigma^2 \left( \frac{\partial^2 G}{\partial x^2} + \frac{\partial^2 G}{\partial y^2} \right)$$

$$\text{NLoG}_{2D}(x, y, \sigma) = -\frac{1}{2\pi\sigma^2} \left( 2 - \frac{x^2 + y^2}{\sigma^2} \right) e^{-\frac{x^2+y^2}{2\sigma^2}}$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-14.png" alt="2D Filter Operators: Laplacian, Gaussian, LoG, NLoG" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 14: 3D surface plots of 2D filter operators: Laplacian ($\nabla^2$), Gaussian ($n_\sigma$), LoG ($\nabla^2 n_\sigma$), and Normalized NLoG ($\sigma^2 \nabla^2 n_\sigma$).</em></figcaption>
  </div>
</figure>

Convolving an image with NLoG filters across multiple scale levels produces a 3D **Scale-Space Volume**:

$$V(x, y, \sigma) = I(x, y) * \left[ \sigma^2 \nabla^2 G(x, y, \sigma) \right]$$

Local extrema points $(x^*, y^*, \sigma^*)$ extracted within this 3D volume represent the true locations and scales of all image blobs.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-15.png" alt="Scale-Space Volume Visualization" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 15: Scale-Space representation $S(x,y,\sigma_0) \dots S(x,y,\sigma_3)$ on the falling man image. Increasing $\sigma$ reduces resolution and smoothes out fine details.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-16.png" alt="Characteristic Scale Peak on Textured Region" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 16: NLoG response across scale at the eye region. A prominent peak occurs at scale $\sigma_1$, identifying its Characteristic Scale (Lindeberg 1994).</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-17.png" alt="No Extremum on Flat Homogeneous Region" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 17: NLoG response across scale on a flat background point. Lacking a strong extremum, no blob is detected.</em></figcaption>
  </div>
</figure>

---

## 4. SIFT Detector

Developed by David Lowe, the **SIFT (Scale-Invariant Feature Transform)** detector introduces key engineering innovations to make scale-space blob detection computationally efficient, fast, and robust to noise.

### 4.1 Fast NLoG Approximation: Difference of Gaussians (DoG)

Computing 2D NLoG convolutions at every scale level is computationally expensive. Lowe demonstrated that subtracting two adjacent Gaussian-smoothed images in scale-space—known as the **Difference of Gaussians (DoG)** operator—provides a close mathematical approximation to NLoG:

$$\text{DoG}(x, y, \sigma) = S(x, y, k\sigma) - S(x, y, \sigma) = I(x, y) * \left[ G(x, y, k\sigma) - G(x, y, \sigma) \right]$$

From the heat diffusion equation, the limit relationship yields:

$$\frac{\partial G}{\partial \sigma} = \lim_{\Delta\sigma \to 0} \frac{G(x,y,\sigma + \Delta\sigma) - G(x,y,\sigma)}{\Delta\sigma}$$

$$\sigma \nabla^2 G = \frac{\partial G}{\partial \sigma} \approx \frac{G(x,y,k\sigma) - G(x,y,\sigma)}{(k-1)\sigma}$$

Multiplying both sides by $\sigma$ directly relates DoG to the $\sigma$-normalized NLoG:

$$G(x,y,k\sigma) - G(x,y,\sigma) \approx (k-1) \cdot \left[ \sigma^2 \nabla^2 G \right] = (k-1) \cdot \text{NLoG}$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-18.png" alt="Comparison between NLoG and DoG Curves" style="display:flex; border-radius: 5px; justify-content: center; width: 450px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 18: Close mathematical alignment between the exact scale-normalized Laplacian (NLoG) curve and the Difference of Gaussians (DoG) approximation ($DoG \approx (s-1)\text{NLoG}$).</em></figcaption>
  </div>
</figure>

By simply taking pixel-wise differences between adjacent Gaussian-blurred images, the expensive NLoG calculation is replaced by efficient image subtraction.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-19.png" alt="Building the DoG Scale-Space Pyramid" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 19: Input image $I(x,y)$ passed through Gaussian scale-space, followed by adjacent scale subtractions to build the DoG volume (Lowe 2004).</em></figcaption>
  </div>
</figure>

### 4.2 3D Extremum Search and Filtering Weak Keypoints

To detect stable keypoints in the DoG volume:

1. A $3 \times 3 \times 3$ cubic window is centered over each sample point in the DoG stack.
2. The pixel's value is compared against its 8 spatial neighbors at the current scale, 9 neighbors at the scale above, and 9 neighbors at the scale below (a total of **26 neighbors**).
3. If the central pixel is strictly greater than or less than all 26 neighbors, it is designated as a keypoint candidate.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-20.png" alt="3D Local Extremum Search in 26 Neighborhood" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 20: 3D local extremum check comparing a central pixel against 26 neighbors in a $3 \times 3 \times 3$ scale-space grid.</em></figcaption>
  </div>
</figure>

**Pruning Low-Contrast and Edge Responses:** Candidate keypoints with low contrast are discarded by thresholding DoG value magnitude. Additionally, unstable keypoints along edges are eliminated by checking the eigenvalue ratio of the local 2D Hessian matrix. The remaining points form the finalized set of stable SIFT interest points.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-21.png" alt="Selection of Stable SIFT Keypoints" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 21: Removal of weak extrema and edge responses, leaving stable SIFT keypoint circles with scale-dependent radii (Lowe 2004).</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-22.png" alt="Detected SIFT Keypoints on God of War Cover" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 22: SIFT keypoints visualized as scale-proportional circles ($r \propto \sigma^*$) on a PS2 game cover.</em></figcaption>
  </div>
</figure>

### 4.3 Achieving Scale and Rotation Invariance

#### 1. Scale Invariance
Changes in camera distance alter object magnification, causing DoG peak extrema to shift to different Characteristic Scales ($\sigma^*$). The ratio of these scales ($\frac{\sigma_1^*}{\sigma_2^*}$) reflects the physical magnification ratio. SIFT normalizes keypoint regions by resampling local patches according to their characteristic scale radius before descriptor extraction.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-23.png" alt="Ratio of Blob Sizes via Characteristic Scale" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 23: Characteristic scale ratio ($\frac{\sigma_1^*}{\sigma_2^*}$) directly measures the relative scale difference between observations (Mikolajczyk 2001).</em></figcaption>
  </div>
</figure>

#### 2. Rotation Invariance and Principal Orientation
A square patch window is constructed around each keypoint at its characteristic scale.

1. For every pixel in the window, horizontal ($I_x$) and vertical ($I_y$) partial derivatives are computed to yield gradient magnitude ($m$) and orientation ($\theta$):

$$m(x,y) = \sqrt{I_x^2 + I_y^2} \quad \text{and} \quad \theta(x,y) = \tan^{-1}\left( \frac{I_y}{I_x} \right)$$

2. To gain immunity against lighting changes, gradient magnitudes are discarded and only orientation angles ($\theta$) are accumulated.
3. Orientation angles ($0^\circ - 360^\circ$) are binned into a 36-bin **Gradient Orientation Histogram**.
4. The dominant peak in the histogram defines the keypoint's **Principal Orientation ($\theta_{\text{principal}}$)**.
5. During matching, the patch is rotated backward by $\theta_{\text{principal}}$, aligning it upright (**North**). This eliminates in-plane rotation effects.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-24.png" alt="Principal Orientation Histogram Calculation" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 24: (Left) Image gradient orientation vectors in normalized window. (Right) 36-bin orientation histogram and peak selection.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-25.png" alt="Principal Orientation Alignment on Rotated CD Cover" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 25: Orientation assignment on a rotated CD cover, enabling patch re-orientation to a canonical upright view.</em></figcaption>
  </div>
</figure>

---

## 5. SIFT Descriptor

Once scale and orientation effects are normalized, a compact and distinctive local descriptor vector must be generated from the upright patch.

### 5.1 Mathematical Construction of the SIFT Descriptor

1. A pixel grid is established over the normalized, oriented keypoint patch.
2. Only gradient orientation angles ($\theta$) are evaluated to maintain illumination insensitivity.
3. The patch area is divided into 4 non-overlapping spatial quadrants ($2 \times 2$).
4. An 8-bin local orientation histogram ($0^\circ, 45^\circ, 90^\circ, \dots, 315^\circ$) is computed independently for each quadrant.
5. The 4 quadrant histograms are concatenated into a unified vector.
6. In Lowe's standard implementation, a $16 \times 16$ pixel region is partitioned into a $4 \times 4$ array of sub-regions, generating an 8-bin histogram per sub-region. This yields the famous **128-dimensional SIFT Descriptor vector** ($16 \times 8 = 128$).

```
 Grid Structure                        4 Quadrant Histograms
 ┌──────────┬──────────┐  
 │          │          │                Local Hist 1 ──┐
 │ Quadrant │ Quadrant │                Local Hist 2 ──┼──► Concatenate ──► [ SIFT Descriptor Vector ]
 │    1     │    2     │                Local Hist 3 ──┼──►   (128D Invariant Signature)
 ├──────────┼──────────┤                Local Hist 4 ──┘
 │ Quadrant │ Quadrant │
 │    3     │    4     │
 └──────────┴──────────┘
```

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-26.png" alt="SIFT Descriptor Construction" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 26: SIFT Descriptor creation: Oriented patch divided into spatial sub-grids, generating local orientation histograms concatenated into a 128D vector.</em></figcaption>
  </div>
</figure>

### 5.2 Distance Metrics for Matching SIFT Descriptors ($H_1, H_2$)

1. **L2 Distance (Euclidean Distance):**
   Square root of the sum of squared differences between descriptor entries. Values closer to zero indicate strong similarity:

   $$D(H_1, H_2) = \sqrt{\sum_{k} \left( H_1[k] - H_2[k] \right)^2}$$

2. **Normalized Correlation:**
   Mean-centered descriptor correlation scaled by total energy. A value of 1.0 indicates perfect linear agreement:

   $$D(H_1, H_2) = \frac{\sum_{k} (H_1[k] - \mu_1)(H_2[k] - \mu_2)}{\sqrt{\sum_{k} (H_1[k] - \mu_1)^2 \sum_{k} (H_2[k] - \mu_2)^2}} \quad \text{where} \quad \mu = \frac{1}{N} \sum_{k} H[k]$$

3. **Intersection Metric:**
   Sum of minimum values across corresponding histogram bins, representing overlap area:

   $$D(H_1, H_2) = \sum_{k} \min\left( H_1[k], H_2[k] \right)$$

### 5.3 SIFT Matching Examples and Applications

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-27.png" alt="SIFT Matching across Scale Changes" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 27: SIFT matches established across large scale changes (Donnie Darko DVD and God of War covers).</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-28.png" alt="SIFT Matching under Rotation" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 28: Robust SIFT matches under $45^\circ$, $90^\circ$, and inverted $180^\circ$ CD cover rotations.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-29.png" alt="SIFT Matching under Clutter and Occlusion" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 29: Successful object retrieval in cluttered, partially occluded CD pile scenes using SIFT.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-30.png" alt="Mountain Landscape SIFT Point Matching" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 30: Automatic keypoint correspondence matching across two mountain landscape photos (Autostitch).</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-31.png" alt="Image Warping and Panorama Stitching" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 31: Geometric image warping and seamless panorama stitching using matched SIFT points.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-32.png" alt="Large Scale Photo Collage Creation" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 32: Large indoor/outdoor collage synthesized from 30 window photos via SIFT matching (Nomura 2007).</em></figcaption>
  </div>
</figure>

### 5.4 Limitations of SIFT and 3D Viewpoint Sensitivity

While SIFT produces hundreds of stable matches for 2D planar surfaces undergoing rotation, scaling, and occlusion, **SIFT breaks down when applied to 3D non-planar objects**.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/sift-detector-33.png" alt="3D Viewpoint Breakdown in SIFT" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 33: Sensitivity of SIFT to 3D viewpoint changes: $0^\circ$ change (100% matching), $30^\circ$ change (sharp drop in matches), $90^\circ$ change (complete breakdown of matching).</em></figcaption>
  </div>
</figure>

As camera viewpoint changes relative to a 3D object, out-of-plane rotations alter local appearance due to 3D self-occlusion and perspective deformation. Empirical studies demonstrate:

- **At a $30^\circ$ viewpoint shift:** The number of matched SIFT keypoints drops drastically.
- **At a $90^\circ$ viewpoint shift:** Keypoint correspondences degrade completely, yielding zero valid matches.

> **Conclusion:** SIFT is reliable primarily for 2D planar scenes or small 3D viewpoint variations.

---

## 6. Technical Summary Matrix

| Module / Topic | Core Equation | Target Output / Value | Problem Solved | Fundamental Limitation |
| :--- | :--- | :--- | :--- | :--- |
| **Interest Point** | Circular patch (Blobs) | Spatial location ($x, y$), scale radius ($\sigma$), and orientation ($\theta$). | Resolves sliding edge ambiguity (aperture problem) and corner sparsity. | Homogeneous, flat, untextured image regions. |
| **Blob Detection** | $\text{NLoG} = \sigma^2 \nabla^2 G$ | Characteristic Scale ($\sigma^*$) and location ($x^*, y^*$). | Detects objects across scale via 3D scale-space extrema search. | High computational cost of multi-scale 2D Gaussian convolutions. |
| **SIFT Detector** | $\text{DoG} = S(k\sigma) - S(\sigma)$ | Scale and rotation normalized keypoints. | Fast NLoG approximation via DoG and principal orientation assignment. | Noisy and unstable extrema candidates across adjacent scales. |
| **SIFT Descriptor** | Vector Concatenation | 128-dimensional invariant visual signature. | Enables stable matching under occlusion, rotation, and illumination. | Total breakdown on 3D objects with $30^\circ - 90^\circ$ viewpoint shifts. |
