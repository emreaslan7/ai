# Homography Estimation, RANSAC, Warping and Blending

<!-- toc -->

## 1. Computing Homography

### 1.1 Role in Image Stitching

When a camera rotates around its optical center to capture images from different angles, all resulting image planes (e.g., $\Pi_1, \Pi_2, \Pi_3$) share the identical projection center (pinhole). Consequently, points across these image planes are directly linked by homography matrices. By cascading homographies via composition, all images can be seamlessly aligned onto a single reference plane ($\Pi_p$).

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-01.png" alt="Image Planes Captured from Shared Projection Center" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 1: Image planes (Π₁, Π₂, Π₃) captured by rotating around a pinhole and their homographic projections onto common reference plane (Πₚ).</em></figcaption>
  </div>
</figure>

### 1.2 Conditions of Homography Validity

Homography-based image alignment is mathematically valid in three primary scenarios:

1. **Same Viewpoint (Pure Rotation):** The camera rotates strictly around its optical center without translation. In this case, homography is exact regardless of the 3D scene depth structure.
2. **Planar Scenes:** Even if the camera translates to different positions, homography holds if the scene object itself is planar in 3D space (e.g., a wall painting or building facade).
3. **Plane at Infinity:** When the scene is extremely distant compared to camera displacement (e.g., distant mountain landscapes), the scene behaves as a plane at infinity, preserving homography validity.

> **Invalid Case (Parallax Artifacts):** When a scene is close to the camera, contains complex 3D depth variations, and the camera translates, homography fails, giving rise to parallax errors.

### 1.3 Direct Linear Transform (DLT)

Let $H$ be the $3 \times 3$ homography matrix mapping a point $p_s[x_s, y_s, 1]^T$ in the source image to point $p_d[x_d, y_d, 1]^T$ in the destination image:

$$p_d \equiv H \cdot p_s$$

$$\begin{bmatrix} \tilde{x}_d \\ \tilde{y}_d \\ \tilde{z}_d \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x_s \\ y_s \\ 1 \end{bmatrix}$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-02.png" alt="Homography Point Correspondence Mapping" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 2: Homography mapping between corresponding point p⛶ in Source Image and p_d in Destination Image.</em></figcaption>
  </div>
</figure>

Expanding this linear system and performing homogeneous normalization ($x_d = \tilde{x}_d / \tilde{z}_d$ and $y_d = \tilde{y}_d / \tilde{z}_d$), each point match provides two linear equations:

$$x_d = \frac{h_{11}x_s + h_{12}y_s + h_{13}}{h_{31}x_s + h_{32}y_s + h_{33}}$$

$$y_d = \frac{h_{21}x_s + h_{22}y_s + h_{23}}{h_{31}x_s + h_{32}y_s + h_{33}}$$

Rearranging terms with respect to unknown matrix entries $h_{ij}$:

$$x_s h_{11} + y_s h_{12} + h_{13} - x_d x_s h_{31} - x_d y_s h_{32} - x_d h_{33} = 0$$

$$x_s h_{21} + y_s h_{22} + h_{23} - y_d x_s h_{31} - y_d y_s h_{32} - y_d h_{33} = 0$$

Since each point match provides 2 independent constraints and homography has 8 degrees of freedom, a **minimum of 4 point correspondences (minimum 4 pairs)** is required to solve $H$.

### 1.4 Constrained Least Squares Estimation

In practical settings, more than 4 point matches ($N > 4$) are utilized to suppress noise, yielding an overdetermined linear system. Stacking equations for all $N$ pairs produces a $2N \times 9$ coefficient matrix $A$:

$$A \cdot h = 0$$

where $h = [h_{11}, h_{12}, h_{13}, h_{21}, h_{22}, h_{23}, h_{31}, h_{32}, h_{33}]^T$. To prevent the trivial solution $h = 0$, we enforce the scale constraint $\|h\|^2 = 1$.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-03.png" alt="Matrix A Stacking and Constrained Least Squares Formulation" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 3: Stacking N point correspondences into 2N x 9 matrix A and constrained least-squares formulation ||h||² = 1.</em></figcaption>
  </div>
</figure>

We formulate the optimization problem to minimize $\|A \cdot h\|^2$ subject to $\|h\|^2 = 1$:

$$\min_{h} h^T A^T A h \quad \text{subject to} \quad h^T h = 1$$

Adding a Lagrange multiplier $\lambda$ yields the Lagrangian loss function:

$$\mathcal{L}(h, \lambda) = h^T A^T A h - \lambda (h^T h - 1)$$

Taking partial derivatives with respect to $h$ and setting to zero leads to the standard Eigenvalue problem:

$$A^T A h = \lambda h$$

> **Optimal Solution:** The parameter vector $h$ minimizing the error corresponds to the **eigenvector associated with the smallest eigenvalue of $A^T A$**. Computing Singular Value Decomposition (SVD) $A = U \Sigma V^T$, $h$ is given by the last column of $V$. Reshaping $h$ into $3 \times 3$ yields homography matrix $H$.

---

## 2. Dealing with Outliers: RANSAC

### 2.1 The Outlier Problem

Feature detectors like SIFT identify matches based purely on local descriptor similarity. Repetitive patterns, reflections, or noise inevitably introduce false matches (outliers) that do not represent identical 3D points.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-04.png" alt="Inliers vs Outliers in Feature Matching" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 4: Genuine point matches (Inliers - Green lines) versus false matches (Outliers - Red lines) across images.</em></figcaption>
  </div>
</figure>

```
  [Inliers (Valid Matches)]               [Outliers (False Matches)]
     Corresponding 3D points                 Incorrect pairings caused by
     in shared scene space                   descriptor similarity or noise
```

If outliers are included in standard least-squares estimation, the estimated transformation matrix is severely distorted. Outliers must be rejected before computing the final homography.

### 2.2 RANSAC (Random Sample Consensus) Algorithm

RANSAC is a robust consensus algorithm capable of estimating accurate model parameters even when outliers exceed 50% of the dataset.

RANSAC execution steps for homography estimation:

1. Randomly select a minimal subset of 4 point matches ($s = 4$).
2. Compute candidate homography matrix $H$ from these 4 points via DLT.
3. Project all data points using candidate $H$ and measure reprojection error. Matches with reprojection error below threshold $\epsilon$ are classified as **Inliers**, yielding consensus score $M$.
4. Repeat steps 1–3 for $N$ iterations.
5. Select the candidate matrix $H$ with the highest consensus score $M$ as the winning model.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-05.png" alt="Least Squares Fitting vs RANSAC First Iteration" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 5: Standard Least Squares fitting (severely biased by outliers, Inliers: 2) vs. RANSAC Iteration 1 (Inliers: 4).</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-06.png" alt="RANSAC Winning Consensus Iteration" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 6: RANSAC Iteration i - Achieving maximum consensus (Inliers: 20) once the optimal model is sampled.</em></figcaption>
  </div>
</figure>

> **Model Refinement:** After RANSAC selects the winning model, all identified inliers ($M$ points) are pooled together. Constrained Least Squares is re-executed over the full inlier set to produce a refined, highly accurate homography matrix $H$.

---

## 3. Image Warping and Blending

After computing homography $H$, geometric warping and photometric blending operations assemble individual images into a seamless panorama.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-07.png" alt="Image Warping Fundamental Concept" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 7: Image Warping: Bending input image f(x,y) onto target plane g(x,y) via coordinate operator T(x,y).</em></figcaption>
  </div>
</figure>

### 3.1 Forward Warping and Hole Artifacts

In forward warping, transformation $H$ is applied to each pixel coordinate $(x_s, y_s)$ in the source image to compute destination coordinate $(x_d, y_d)$, writing source pixel color to that target location.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-08.png" alt="Forward Warping and Grid Holes" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 8: Forward Warping: Source pixels map to non-integer destination grid locations, leaving unassigned black holes.</em></figcaption>
  </div>
</figure>

Forward warping suffers from two major drawbacks:
1. **Non-integer Coordinates:** Transformed coordinates rarely align with integer pixel grid centers in the output image.
2. **Holes and Gaps:** Geometric expansion leaves target pixels unmapped by any source pixel, producing unassigned black holes.

### 3.2 Backward Warping

To eliminate hole artifacts, backward warping is performed:

1. Transform the 4 corners of the source image using forward homography to determine output bounding box dimensions.
2. Iterate through every integer pixel coordinate $(x_d, y_d)$ within the output canvas.
3. Apply inverse homography ($H^{-1}$) to locate source coordinate $(x_s, y_s)$.
4. Sample pixel color from the source image at $(x_s, y_s)$ using **Bilinear Interpolation** or **Nearest Neighbor**.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-09.png" alt="Backward Warping Scheme" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 9: Backward Warping: Mapping from output pixel back to source image via H⁻¹ and sampling color via interpolation.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-10.png" alt="Multiple Image Bounding Box Calculation" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 10: Computing output canvas bounding box by projecting image corners onto common reference plane.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-11.png" alt="Inverse Homography Fetching from Source Images" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 11: Inverse Homographies (H₁₂, H₃₂) sampling pixel data from original source images into reference canvas.</em></figcaption>
  </div>
</figure>

```
  [Forward Warping]  (x, y)   ──► H   ──► (x', y')   (Leaves gaps and unassigned holes)
  [Backward Warping] (x', y') ──► H^-1 ──► (x, y)     (Seamless, gap-free output)
```

Because every pixel in the output canvas is back-projected and sampled, backward warping guarantees a completely gap-free composite image.

### 3.3 Image Blending and Seam Artifacts

Even when images are aligned with geometric precision, directly overlaying them creates sharp seam boundaries (*hard seams*).

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-12.png" alt="Direct Image Overlay Hard Seam Formation" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 12: Direct overlay of images (Hard overlay / step-function weights w₁, w₂) producing sharp visible seams.</em></figcaption>
  </div>
</figure>

Seams arise due to two primary optical factors:
1. **Exposure and Illumination Variations:** Automatic camera exposure adjustments or dynamic ambient lighting changes between shots.
2. **Vignetting Effects:** Lens falloff causing pixel brightness to decrease near image boundaries compared to the center.

Human vision is acutely sensitive to intensity steps as small as 1 gray level across smooth regions. Simple pixel averaging softens transition boundaries but fails to eliminate seams.

### 3.4 Weighted Blending

To eliminate seam lines, pixel weights are assigned based on spatial proximity to image centers. The blended pixel intensity ($I_{\text{blend}}$) is computed using smooth weight matrices $w_1$ and $w_2$:

$$I_{\text{blend}} = \frac{w_1 I_1 + w_2 I_2}{w_1 + w_2}$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-13.png" alt="Weighted Blending Linear Ramps" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 13: Smooth ramp weight functions (w₁, w₂) and weighted blending equation formulation.</em></figcaption>
  </div>
</figure>

### 3.5 Distance Transform-Based Blending

Optimal blending weights are computed using the **Distance Transform** (e.g., MATLAB `bwdist`):

1. The weight of each pixel is proportional to its Euclidean distance from the nearest image boundary.
2. Pixels near the center receive higher weight ($w$), reflecting higher optical quality and lower vignetting falloff. Boundary pixel weights decay smoothly to zero.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-14.png" alt="Distance Transform Weighting Maps" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 14: Alpha weight maps (w₁, w₂, w₃) generated via Distance Transform for Images 1, 2, and 3.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-15.png" alt="Raw Overlay vs Distance Transform Blended Panorama" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 15: (Top) Raw overlay with visible exposure boundary steps vs. (Bottom) Distance transform blended seamless panorama.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-16.png" alt="Multi-Image Panoramic Mosaic Alignment" style="display:flex; border-radius: 5px; justify-content: center; width: 650px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 16: Panoramic mosaic generated from 6 source images via pairwise homographies, backward warping, and distance blending.</em></figcaption>
  </div>
</figure>

Distance transform blending spreads intensity transitions smoothly across overlap regions, producing high-resolution panoramas free of visible seam artifacts.
