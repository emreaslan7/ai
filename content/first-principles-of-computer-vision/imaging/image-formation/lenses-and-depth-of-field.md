# Lens Systems and Depth of Field

<!-- toc -->

## 1. Why Lenses?

Pinhole cameras produce sharp images, but the extremely small aperture collects very little light — the Flatiron building example required a **12-second exposure**. Lenses solve this problem by refracting light from a wide aperture to converge at a single point, increasing brightness while preserving the perspective model.

> **Fundamental Trade-off:** Lenses collect more light but introduce a finite depth of field — only one plane is perfectly in focus.

## 2. Gaussian Lens Law

For a thin lens, the relationship between the object distance ($o$), image distance ($i$), and focal length ($f$) is given by the **Gaussian Lens Law**:

$$
\frac{1}{i} + \frac{1}{o} = \frac{1}{f}
$$

```mermaid
flowchart LR
    A["Object<br/>Distance o"] --> B["Thin Lens<br/>Focal Length f"]
    B --> C["Image<br/>Distance i"]
    D["1/f = 1/i + 1/o"] -.- B
    
    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#16213e,stroke:#4cc9f0,color:#fff
    style C fill:#0f3460,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#888,color:#888
```

**Numerical Example:** With a lens of $f = 50$mm focused on an object at $o = 300$mm:

$$
\frac{1}{i} = \frac{1}{50} - \frac{1}{300} = \frac{6 - 1}{300} = \frac{5}{300}
$$

$$
i = 60 \text{ mm}
$$

The image forms 60 mm behind the lens.

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/lenses-and-depth-of-field-01.png" alt="Gaussian Lens Law Diagram" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Similar triangles derive the Gaussian Lens Law equations.</em></figcaption>
</div>
</figure>

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/lenses-and-depth-of-field-06.png" alt="Measuring Focal Length" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Measuring focal length with a street lamp in practice.</em></figcaption>
</div>
</figure>

### 2.1 Aperture and f-Number

The light-gathering capacity of a lens is determined by the aperture diameter ($D$). The **f-number** ($N$) is defined as:

$$
N = \frac{f}{D}
$$

| Aperture | f-Number | Light Collected | Depth of Field |
|----------|----------|-----------------|----------------|
| Wide open | Low $N$ (e.g., $f/1.4$) | High | Shallow |
| Stopped down | High $N$ (e.g., $f/16$) | Low | Deep |

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/lenses-and-depth-of-field-02.png" alt="Nikon Aperture Blades" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Aperture blades create different f-number openings.</em></figcaption>
</div>
</figure>

### 2.2 The Tissue Box Experiment

A fascinating and counter-intuitive observation: **covering half of a lens does not break or defocus the image.** It only reduces the light reaching the sensor, darkening the image. Every unblocked portion of the lens continues to project the entire scene onto the focal plane.

> **Why?** Each point on the lens receives light from all scene points within its field of view. Blocking part of the lens reduces the number of rays but does not change their geometric paths — the entire scene is still projected, just dimmer.

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/lenses-and-depth-of-field-07.png" alt="Tissue Box Camera" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>A tissue box camera demonstrates the lens principle.</em></figcaption>
</div>
</figure>

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/lenses-and-depth-of-field-03.png" alt="Blocking the Lens" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Blocking half the lens only darkens the image.</em></figcaption>
</div>
</figure>

### 2.3 Zoom

Zoom is the process of changing the magnification by moving lens elements within a multi-lens system. This changes the effective focal length without physically swapping lenses.

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/lenses-and-depth-of-field-08.png" alt="Two Lens Zoom System" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Two-lens system enables zoom by moving elements.</em></figcaption>
</div>
</figure>

## 3. Defocus Blur and Depth of Field

A lens system perfectly focuses only a single **focal plane** at a specific sensor position. Points outside this plane form a **blur circle** (circle of confusion) on the image plane.

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/lenses-and-depth-of-field-04.png" alt="Depth of Field Example" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Depth of field varies with aperture size in practice.</em></figcaption>
</div>
</figure>


### 3.1 The Blur Circle

Using similar triangles, the diameter of the blur circle ($b$) is related to the aperture diameter ($D$):

$$
\frac{b}{D} = \frac{|i' - i|}{i'}
$$

Where $i'$ is the image distance of the out-of-focus point, and $i$ is the sensor distance.

```mermaid
flowchart LR
    subgraph InFocus["In Focus"]
        A1["Scene Point on Focal Plane"] --> B1["Lens"] --> C1["Sharp Point on Sensor"]
    end
    subgraph OutOfFocus["Out of Focus"]
        A2["Scene Point off Focal Plane"] --> B2["Lens"] --> C2["Blur Circle on Sensor"]
    end
    
    style A1 fill:#16213e,stroke:#4cc9f0,color:#fff
    style B1 fill:#1a1a2e,stroke:#4cc9f0,color:#fff
    style C1 fill:#0f3460,stroke:#4cc9f0,color:#fff
    style A2 fill:#16213e,stroke:#e94560,color:#fff
    style B2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style C2 fill:#0f3460,stroke:#e94560,color:#fff
```

This equation proves that the blur circle diameter is **directly proportional to the aperture diameter** — wider apertures produce more defocus blur.

### 3.2 Depth of Field (DoF)

The **Depth of Field** is the range of depths over which the blur circle diameter remains smaller than the pixel size ($C$). If $b < C$, the image is perceived as "sharp."

$$
\text{DoF} \propto \frac{N \cdot C \cdot o^2}{f^2}
$$

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/lenses-and-depth-of-field-09.png" alt="Depth of Field Depth Limits" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Depth of field limits where blur stays below pixel size.</em></figcaption>
</div>
</figure>

### 3.3 Hyperfocal Distance

The **hyperfocal distance** ($H$) is the focus distance at which everything from that point to infinity appears acceptably sharp:

$$
H = \frac{f^2}{N \cdot C} + f
$$

Smartphone cameras strategically use this parameter — their small sensors and short focal lengths produce a very large hyperfocal distance, ensuring nearly everything is in focus without active focusing.

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/lenses-and-depth-of-field-05.png" alt="Hyperfocal Distance Diagram" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Hyperfocal distance ensures sharpness from H to infinity.</em></figcaption>
</div>
</figure>

### 3.4 The Critical Trade-off

| Scenario | Aperture | Light | Exposure Time | Depth of Field |
|----------|----------|-------|---------------|----------------|
| Bright, shallow DoF | Wide ($N$ low) | High | Short | Shallow |
| Dark, deep DoF | Narrow ($N$ high) | Low | Long | Deep |

<figure style="display:flex; justify-content: center;">
<div>
<img src="../../../img/first-principles-of-computer-vision/lenses-and-depth-of-field-10.png" alt="Aperture DOF vs Brightness" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
<figcaption style="margin-top: 0.5em; text-align: center;"><em>Wider aperture increases blur but gathers more light.</em></figcaption>
</div>
</figure>

There is **no free lunch** in optical design — every gain in one dimension comes at a cost in another.

---

## Summary

- **Lenses** increase light collection but introduce finite depth of field.
- **Gaussian Lens Law**: $1/i + 1/o = 1/f$ governs thin lens behavior.
- **f-Number** $N = f/D$ quantifies aperture size and directly affects light and DoF.
- **Blur circle** $b/D = |i' - i|/i'$ proves defocus is proportional to aperture.
- **Hyperfocal distance** $H = f^2/(N \cdot C) + f$ enables strategic focus optimization.
- The aperture trade-off (light vs. DoF) is fundamental and unavoidable.
