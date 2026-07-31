# Advanced Optical Systems: Aberrations, Wide-Angle Imaging, and Biological Eyes

<!-- toc -->

## 1. Lens Aberrations

Even perfect lenses produce unwanted effects called **aberrations** due to the nature of light. These are physical limitations, not manufacturing defects.

### 1.1 Vignetting

Vignetting is the darkening of image corners caused by:

1. The lens body mechanically blocking oblique rays.
2. A reduction in **solid angle** at the periphery of the image field.

The result is a gradual fall-off in brightness from the center to the corners of the image.

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-01.png" alt="Vignetting Ray Diagram" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Ray diagram showing mechanical blockage of oblique rays in multi-lens systems.</em></figcaption>
</div>
</figure>

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-02.png" alt="Vignetting Example" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Vignetting effect on a flat white surface and a natural scene.</em></figcaption>
</div>
</figure>

### 1.2 Chromatic Aberration

The refractive index of glass depends on the wavelength ($\lambda$) of light. In the visible spectrum (400 nm — 700 nm), **blue light (400 nm) bends more than red light (700 nm)**. This causes different colors to focus at different planes, producing color fringing at object edges.

```mermaid
flowchart LR
    A["White Light<br/>400-700 nm"] --> B["Lens"]
    B --> C["Blue Focus<br/>(shorter focal length)"]
    B --> D["Red Focus<br/>(longer focal length)"]
    C --> E["Color Fringing at Edges"]
    D --> E
    
    style A fill:#1a1a2e,stroke:#fff,color:#fff
    style B fill:#16213e,stroke:#4cc9f0,color:#fff
    style C fill:#1a1a2e,stroke:#4361ee,color:#4361ee
    style D fill:#1a1a2e,stroke:#e94560,color:#e94560
    style E fill:#0f3460,stroke:#f72585,color:#fff
```

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-03.png" alt="Chromatic Aberration" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Chromatic aberration and edge fringing caused by different wavelengths bending differently.</em></figcaption>
</div>
</figure>

### 1.3 Geometric Distortions

**Radial distortion** (barrel distortion) causes the image to bulge outward. These effects can be corrected in computer vision software through inverse mapping — a calibration process that models the distortion parameters and inverts them.

| Distortion Type | Effect | Visual |
|----------------|--------|--------|
| **Barrel (Fıçı)** | Lines bow outward from center | 👁️ Wide-angle look |
| **Pincushion (İğne Yastığı)** | Lines bow inward toward center | 🔍 Telephoto look |

> **Key Insight:** Distortion is deterministic and correctable — knowing the lens model allows precise geometric rectification.

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-04.png" alt="Geometric Distortion Types" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Radial (barrel/pincushion) and tangential geometric distortion diagram from lens imperfections.</em></figcaption>
</div>
</figure>

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-05.png" alt="Distortion Correction" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Barrel-distorted corridor photo before and after software rectification.</em></figcaption>
</div>
</figure>

## 2. Wide-Angle and Catadioptric Imaging Systems

These systems are designed to overcome the limitations of standard perspective projection and are strategically important in security and robotics.

### 2.1 Fisheye Lenses

Fisheye lenses use meniscus elements to achieve extreme light bending. The **single viewpoint constraint** is critical for software rectification — all rays must appear to converge at a single optical center for the image to be mathematically unwrapped.

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-06.png" alt="Fisheye Lens Design" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Fisheye lens design using meniscus elements for extreme light bending.</em></figcaption>
</div>
</figure>

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-07.png" alt="Fisheye Hemispherical Image" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Fisheye lens and the 180-degree hemispherical image it captures.</em></figcaption>
</div>
</figure>

### 2.2 Catadioptric Systems

Catadioptric systems combine mirrors (catoptric) and lenses (dioptric):

| Type | Mirror Shape | Use Case |
|------|-------------|----------|
| **Telescope** | Parabolic | Collects parallel rays at a single point |
| **Omnidirectional** | Hyperbolic (convex) | Captures 360° panoramic view for surveillance |

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-08.png" alt="Hyperbolic Mirror Ray Diagram" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Ray tracing diagram showing rays reflecting from a hyperbolic mirror converging at a virtual focus.</em></figcaption>
</div>
</figure>

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-09.png" alt="Parabolic Mirror Projection" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Parabolic mirror orthographic projection capturing parallel rays.</em></figcaption>
</div>
</figure>

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-10.png" alt="James Webb Mirror" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>James Webb Space Telescope's massive concave mirror system.</em></figcaption>
</div>
</figure>

### 2.3 Corneal Imaging

The human cornea acts as a convex mirror. Using **limbus detection** and corneal reflection analysis, it is possible to determine what a person is looking at (their retinal image) from a high-resolution photograph taken from outside.

```mermaid
flowchart LR
    A["External Camera"] -->|"High-res photo"| B["Corneal Reflection<br/>Convex mirror"]
    B -->|"Limbus detection"| C["Gaze Direction<br/>Analysis"]
    C -->|"Inverse projection"| D["Retinal Image<br/>(What person sees)"]
    
    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#16213e,stroke:#4cc9f0,color:#fff
    style C fill:#0f3460,stroke:#f72585,color:#fff
    style D fill:#1a1a2e,stroke:#06d6a0,color:#fff
```

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-11.png" alt="Limbus Detection" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Elliptical limbus boundary detection for eye position and orientation.</em></figcaption>
</div>
</figure>

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-12.png" alt="Corneal Reflection Analysis" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Extracting the surrounding scene from corneal reflection and estimating the retinal fovea image.</em></figcaption>
</div>
</figure>

## 3. Biological Eye Designs and Evolution

Eyes in nature represent the evolutionary perfection of image formation principles. A simulation by Nilsson demonstrated that a flat, light-sensitive epithelium could evolve into a complex eye in **just 400,000 generations**.

### 3.1 The Evolutionary Path

```mermaid
flowchart LR
    A["Flat Light-Sensitive Epithelium"] --> B["Curving for Directional Sensitivity"]
    B --> C["Aperture Narrowing for Sharpness"]
    C --> D["Lens Formation for Light Collection"]
    D --> E["Complex Eye"]
    
    style A fill:#1a1a2e,stroke:#4cc9f0,color:#fff
    style B fill:#16213e,stroke:#4cc9f0,color:#fff
    style C fill:#0f3460,stroke:#f72585,color:#fff
    style D fill:#16213e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#06d6a0,color:#fff
```

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-13.png" alt="Eye Evolution Simulation" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Simulation of eye evolution from flat tissue to camera-type eye (Nilsson-Pelger model).</em></figcaption>
</div>
</figure>

### 3.2 Comparative Biology

| Species | Eye Type | Key Feature |
|---------|----------|-------------|
| **Trilobites** (400M years ago) | Compound eye | Thousands of calcite crystal lenses |
| **Human** | Single lens eye | Corneal bending power + crystalline lens accommodation |
| **Scallop** | Multiple mirror eyes | Concave parabolic mirrors (like James Webb Telescope) |

> **Fascinating Fact:** Trilobite eyes used calcite — a mineral that does not soften with age — as their lens material, giving them perfect vision throughout their lifespan.

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-14.png" alt="Primitive Eye Comparison" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Anatomical comparison of primitive eye designs: pit, pinhole, spherical lens, and vertebrate.</em></figcaption>
</div>
</figure>

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-15.png" alt="Trilobite Compound Eye" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Ancient trilobite compound eye fossil with calcite crystal lenses.</em></figcaption>
</div>
</figure>

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-16.png" alt="Scallop Mirror Eye" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Scallop eye with concave parabolic mirror telescopes along its shell edge.</em></figcaption>
</div>
</figure>

### 3.3 The Human Eye and Accommodation

The human eye combines two optical elements:

1. **Cornea** — Provides most of the bending power (refractive index difference between air and tissue).
2. **Crystalline Lens** — A fluid-filled flexible lens that adjusts shape for **accommodation** (focusing at different distances).

As we age, the crystalline lens hardens (presbyopia):

$$
\text{Minimum Focus Distance} \approx 
\begin{cases}
7 \text{ cm} & \text{at age 10} \\
10 \text{ cm} & \text{at age 20} \\
50 \text{ cm} & \text{at age 50+}
\end{cases}
$$

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-17.png" alt="Human Eye Anatomy" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Optical anatomy of the human eye including lens, pupil, fovea, and retinal layers.</em></figcaption>
</div>
</figure>

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-18.png" alt="Accommodation Diagram" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Accommodation diagram: lens bulging when focusing near and flattening for distance.</em></figcaption>
</div>
</figure>

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-19.png" alt="Age vs Focus Distance" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Graph showing near focus point receding as the lens hardens with age.</em></figcaption>
</div>
</figure>

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-20.png" alt="Myopia Correction" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Myopia correction using a diverging (concave) lens.</em></figcaption>
</div>
</figure>

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/advanced-optical-systems-21.png" alt="Hyperopia Correction" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Hyperopia correction using a converging (convex) lens.</em></figcaption>
</div>
</figure>

### 3.4 Scallop Eyes: Nature's Mirror Telescopes

Scallops have hundreds of eyes, each using a **concave parabolic mirror** rather than a lens to focus light. This is the same optical principle used by the James Webb Space Telescope — a remarkable case of convergent evolution between biology and engineering.

---

## Summary

- **Aberrations** (vignetting, chromatic, distortion) are unavoidable physical effects of lens systems.
- **Catadioptric systems** combine mirrors and lenses for specialized imaging (panoramic, telescope).
- **Corneal imaging** allows gaze detection from external photographs.
- **Biological eyes** evolved through a well-understood path and offer diverse optical strategies — pinhole (nautilus), compound (trilobite), refractive (human), and reflective (scallop).
- Whether an ancient trilobite lens or a modern liquid lens, image formation is the central achievement of both biological and technological evolution in understanding the 3D world on a 2D plane.
