# Overview and Image Transformations

<!-- toc -->

## 1. Classification of Image Transformations

Transformations applied in computer vision and image processing fall into two main categories:

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../img/first-principles-of-computer-vision/overview-and-image-transformations-01.png" alt="Image Stitching and Feature Matching Overview" style="display:flex; border-radius: 5px; justify-content: center; width: 650px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 1: (Top) Matching keypoint features across overlapping images. (Bottom) High-resolution panoramic composite generated via geometric transformations and warping.</em></figcaption>
  </div>
</figure>

### 1.1 Image Filtering (Range Transformations)

In image filtering, the pixel spatial coordinates (domain) of the input image remain strictly fixed, while pixel intensity and color values (range) are modified. Pixel processing, linear filtering, and convolution belong to this class. The geometric structure and boundaries of the image remain completely unchanged.

Mathematical formulation:

$$g(x,y) = T_r(f(x,y))$$

where $f(x,y)$ represents the input image, $g(x,y)$ the output image, and $T_r$ the intensity/range transformation operator.

### 1.2 Image Warping (Domain Transformations)

In image warping, operations work directly on the spatial coordinate plane (domain) of the image, altering its geometric shape. Translation, rotation, scaling, affine, and projective transformations belong to this class.

Mathematical formulation:

$$g(x,y) = f(T_d(x,y))$$

where $T_d$ represents the spatial coordinate transformation operator.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../img/first-principles-of-computer-vision/overview-and-image-transformations-02.png" alt="Image Filtering vs Image Warping Comparison" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 2: Image Filtering (Modifies pixel intensity values, coordinates fixed) vs. Image Warping (Modifies pixel spatial coordinates, alters geometry).</em></figcaption>
  </div>
</figure>

```
  [Image Filtering (Range)]               [Image Warping (Domain)]
     f(x, y) ──► T_r ──► g(x, y)             f(x, y) ──► T_d(x, y) ──► g(x', y')
     (Pixel values change,                    (Pixel locations change,
      coordinates fixed)                       shape warped)
```

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../img/first-principles-of-computer-vision/overview-and-image-transformations-03.png" alt="Parametric 2D Transformation Categories" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 3: Parametric 2D Image Warping Transformations (Translation, Rotation, Scaling & Aspect, Affine, Projective, and Barrel distortion).</em></figcaption>
  </div>
</figure>

---

## 2. 2x2 Linear Transformations

The most fundamental geometric operations in two-dimensional space map input pixels to output pixels via a $2 \times 2$ matrix $T$. Given a source pixel $p_1(x_1, y_1)$ and target pixel $p_2(x_2, y_2)$:

$$\begin{bmatrix} x_2 \\ y_2 \end{bmatrix} = \begin{bmatrix} t_{11} & t_{12} \\ t_{21} & t_{22} \end{bmatrix} \begin{bmatrix} x_1 \\ y_1 \end{bmatrix}$$

### 2.1 Scaling (Stretching & Squishing)

To stretch or shrink an image horizontally by factor $a$ and vertically by factor $b$, the coordinate equations are:

$$x_2 = a \cdot x_1, \quad y_2 = b \cdot y_1$$

In matrix form:

$$\begin{bmatrix} x_2 \\ y_2 \end{bmatrix} = \begin{bmatrix} a & 0 \\ 0 & b \end{bmatrix} \begin{bmatrix} x_1 \\ y_1 \end{bmatrix}$$

If scaling matrix $S$ is non-singular (*invertible*, $a \neq 0$ and $b \neq 0$), the inverse matrix $S^{-1}$ allows mapping back from target to source without any loss of geometric information:

$$\begin{bmatrix} x_1 \\ y_1 \end{bmatrix} = S^{-1} \begin{bmatrix} x_2 \\ y_2 \end{bmatrix} = \begin{bmatrix} 1/a & 0 \\ 0 & 1/b \end{bmatrix} \begin{bmatrix} x_2 \\ y_2 \end{bmatrix}$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../img/first-principles-of-computer-vision/overview-and-image-transformations-04.png" alt="2x2 Forward and Inverse Scaling" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 4: Forward Scaling Matrix S and Inverse Scaling Matrix S⁻¹.</em></figcaption>
  </div>
</figure>

### 2.2 2D Rotation

To rotate a point $p_1(x_1, y_1)$ counter-clockwise around the origin by angle $\theta$, we express the initial position using polar coordinates. Let $r$ be distance to origin and $\psi$ the initial angle:

$$x_1 = r \cos \psi, \quad y_1 = r \sin \psi$$

Rotating by angle $\theta$, the new point $p_2(x_2, y_2)$ becomes:

$$x_2 = r \cos(\psi + \theta), \quad y_2 = r \sin(\psi + \theta)$$

Expanding using trigonometric addition formulas:

$$x_2 = r(\cos \psi \cos \theta - \sin \psi \sin \theta) = (r \cos \psi) \cos \theta - (r \sin \psi) \sin \theta$$

$$y_2 = r(\sin \psi \cos \theta + \cos \psi \sin \theta) = (r \cos \psi) \sin \theta + (r \sin \psi) \cos \theta$$

Substituting $x_1$ and $y_1$:

$$x_2 = x_1 \cos \theta - y_1 \sin \theta$$

$$y_2 = x_1 \sin \theta + y_1 \cos \theta$$

Represented in linear matrix form using rotation matrix $R$:

$$\begin{bmatrix} x_2 \\ y_2 \end{bmatrix} = \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix} \begin{bmatrix} x_1 \\ y_1 \end{bmatrix}$$

To invert rotation, inverse matrix $R^{-1}$ is applied. Because rotation matrices are orthogonal ($R^{-1} = R^T$), inversion simply negates the angle:

$$R^{-1} = \begin{bmatrix} \cos \theta & \sin \theta \\ -\sin \theta & \cos \theta \end{bmatrix}$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../img/first-principles-of-computer-vision/overview-and-image-transformations-05.png" alt="2D Rotation and Inverse Rotation Matrices" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 5: Rotation by angle θ around origin (R) and inverse rotation matrix (R⁻¹).</em></figcaption>
  </div>
</figure>

### 2.3 Skew / Shear

Shear transformations convert rectangular regions into parallelograms.

**Horizontal Skew:** Shifts the $x$-coordinate proportionally by factor $m$ of vertical position $y$, leaving $y$ unchanged:

$$\begin{bmatrix} x_2 \\ y_2 \end{bmatrix} = \begin{bmatrix} 1 & m \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ y_1 \end{bmatrix}$$

**Vertical Skew:** Shifts the $y$-coordinate proportionally by factor $m$ of horizontal position $x$, leaving $x$ unchanged:

$$\begin{bmatrix} x_2 \\ y_2 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ m & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ y_1 \end{bmatrix}$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../img/first-principles-of-computer-vision/overview-and-image-transformations-06.png" alt="Horizontal and Vertical Skew Transformations" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 6: Horizontal Skew and Vertical Skew transformation matrices and visual effects.</em></figcaption>
  </div>
</figure>

### 2.4 Mirror / Reflection

**Reflection across Y-axis:** Negates $x$ coordinates while preserving $y$:

$$\begin{bmatrix} x_2 \\ y_2 \end{bmatrix} = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ y_1 \end{bmatrix}$$

**Reflection across line $y = x$ (Diagonal):** Swaps $x$ and $y$ coordinate axes:

$$\begin{bmatrix} x_2 \\ y_2 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} x_1 \\ y_1 \end{bmatrix}$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../img/first-principles-of-computer-vision/overview-and-image-transformations-07.png" alt="Mirror Reflection Transformations" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 7: Reflection across Y-axis (M_y) and diagonal reflection across line y = x (M_xy).</em></figcaption>
  </div>
</figure>

### 2.5 Properties and Limitations of 2x2 Linear Transformations

- **Origin is Invariant:** The origin $(0,0)$ always maps to $(0,0)$.
- **Lines Map to Lines:** Straight lines in input space remain straight lines in output space.
- **Parallelism is Preserved:** Parallel lines remain strictly parallel after transformation.
- **Closed under Composition:** Sequential transformations can be combined into a single matrix multiplication:

$$T_{13} = T_{23} \cdot T_{12}$$

> **Fundamental Limitation of 2x2 Systems (Translation Problem):** Translation ($x_2 = x_1 + t_x$ and $y_2 = y_1 + t_y$), despite being the simplest geometric shift, cannot be expressed as a linear $2 \times 2$ matrix operation because $2 \times 2$ multiplication lacks terms for constant additive offsets $+t_x$ and $+t_y$. To overcome this limitation, coordinates are extended by one dimension into **Homogeneous Coordinates**.

---

## 3. 3x3 Image Transformations

### 3.1 Homogeneous Coordinates

To resolve dimensional constraints and unify translation with linear transformations under a single matrix multiplication, Homogeneous Coordinates are introduced.

A 2D point $p(x,y)$ is represented in homogeneous coordinates by adding a non-zero fictitious scale dimension $\tilde{z}$, forming a 3D point $\tilde{p}(\tilde{x}, \tilde{y}, \tilde{z})$. Mapping back to 2D Cartesian coordinates is defined as:

$$x = \frac{\tilde{x}}{\tilde{z}}, \quad y = \frac{\tilde{y}}{\tilde{z}}$$

Geometrically, the 2D Cartesian plane corresponds to the plane $\tilde{z} = 1$ in 3D homogeneous space. A ray $L$ originating from the origin and passing through $p(x,y,1)$ contains equivalent homogeneous representations of the same 2D point $p(x,y)$.

```
       z_tilde
          ▲          /  Ray L (All points along ray are equivalent)
          │         /
     1.0 ─┼────────• p(x, y, 1)  <-- Projection Plane
          │       /│
          │      / │
          │     /  │
          │    /   │
          └───•────┼─────────► x_tilde
            Origin │
                   ▼ y_tilde
```

Consequently, multiplying homogeneous vector $[x, y, 1]^T$ by any non-zero scale factor $\tilde{z}$ yields $[\tilde{z}x, \tilde{z}y, \tilde{z}]^T$, which represents the identical physical 2D point.

### 3.2 3x3 Representation of Translation

Using homogeneous coordinates, translation becomes a linear $3 \times 3$ matrix multiplication:

$$\begin{bmatrix} x_2 \\ y_2 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ y_1 \\ 1 \end{bmatrix} = \begin{bmatrix} x_1 + t_x \\ y_1 + t_y \\ 1 \end{bmatrix}$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../img/first-principles-of-computer-vision/overview-and-image-transformations-08.png" alt="Translation in Homogeneous Coordinates" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 8: 3x3 translation matrix T in homogeneous coordinates.</em></figcaption>
  </div>
</figure>

All $2 \times 2$ operations (scaling, rotation, skew) are embedded into the upper-left $2 \times 2$ submatrix of $3 \times 3$ homogeneous matrices. A sequence of transformations (e.g., skew, translate, scale, rotate) can thus be concatenated into a single composite $3 \times 3$ matrix applied in a single pass.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../img/first-principles-of-computer-vision/overview-and-image-transformations-09.png" alt="Primary 3x3 Homogeneous Transformation Matrices" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 9: Fundamental 3x3 transformation matrices in homogeneous coordinates (Scaling, Skew, Translation, Rotation).</em></figcaption>
  </div>
</figure>

### 3.3 Affine Transformations

Any $3 \times 3$ homogeneous transformation matrix whose bottom row is fixed to $[0\quad0\quad1]$ belongs to the **Affine Transformation** class:

$$\begin{bmatrix} x_2 \\ y_2 \\ 1 \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} & t_x \\ a_{21} & a_{22} & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ y_1 \\ 1 \end{bmatrix}$$

Affine transformations possess **6 degrees of freedom (DoF)**.

**Properties of Affine Transformations:**
- The origin does not need to map to the origin (translation is supported).
- Lines map to lines.
- Parallel lines remain strictly parallel.
- Closed under composition.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../img/first-principles-of-computer-vision/overview-and-image-transformations-10.png" alt="Affine Transformation Matrix and Geometry" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 10: Affine Transformation matrix (bottom row fixed to [0 0 1]) combining linear deformation and translation.</em></figcaption>
  </div>
</figure>

### 3.4 Projective Transformations (Homography)

When the bottom row of a $3 \times 3$ homogeneous transformation matrix is unconstrained ($[h_{31}, h_{32}, h_{33}]$), the transformation is a **Projective Transformation** or **Homography**:

$$\begin{bmatrix} \tilde{x}_2 \\ \tilde{y}_2 \\ \tilde{z}_2 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x_1 \\ y_1 \\ 1 \end{bmatrix}$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../img/first-principles-of-computer-vision/overview-and-image-transformations-11.png" alt="Homography Projective Transformation Matrix" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 11: Homography (Projective Transformation) matrix H. Bottom row is unconstrained, yielding 8 degrees of freedom.</em></figcaption>
  </div>
</figure>

A projective transformation maps points on a plane $\Pi_1$ through a single projection center (pinhole) onto another plane $\Pi_2$. This models the perspective projection geometry of a camera imaging a planar scene.

**Scale Ambiguity and Degrees of Freedom:**
Due to homogeneous equivalence, multiplying homography matrix $H$ by any non-zero scalar $k$ does not alter final Cartesian coordinates $(x_2, y_2)$. Thus, homography is defined only *up to a scale factor*. Fixing scale via constraint $\sum h_{ij}^2 = 1$ leaves **8 degrees of freedom (DoF)** despite having 9 matrix entries.

**Properties of Projective Transformations:**
- Lines map to lines and composition is closed.
- **Key difference from Affine Transformations:** Parallel lines are **not** preserved. Under perspective projection, parallel lines converge toward vanishing points (e.g., railway tracks converging at the horizon).

---

## 4. Transformation Properties Summary

| Transformation Type | Matrix Size | Degrees of Freedom (DoF) | Preserved Geometric Properties | Bottom Row Constraint |
| :--- | :--- | :--- | :--- | :--- |
| **Linear (2x2)** | $2 \times 2$ | 4 | Origin, Linearity, Parallelism | - |
| **Affine** | $3 \times 3$ | 6 | Linearity, Parallelism | $[0 \quad 0 \quad 1]$ |
| **Projective (Homography)** | $3 \times 3$ | 8 | Linearity | Free (Up to Scale) |

