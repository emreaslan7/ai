# Introduction to Computer Vision

<!-- toc -->

## 1. What is Computer Vision?

Computer vision is not merely a subset of artificial intelligence; it is a profound multidisciplinary engineering and scientific enterprise. It bridges the physical world and symbolic understanding, drawing upon optics, signal processing, electrical engineering, and computer science.

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px;">
    <img src="../../../img/first-principles-of-computer-vision/introduction-to-computer-vision-01.png" style="display:flex; justify-content: center; width: 600px;"alt="Vision pipeline: light source, scene, camera, and Vision Software generating scene description"/>
</div>
<p style="text-align: center; font-size: 14px; color: #888; margin-top: -10px;"><i>Vision pipeline: light source → scene → camera → Vision Software produces a scene description</i></p>

The fundamental challenge lies in transforming raw numerical arrays—pixel data—into a meaningful description of the 3D environment.

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 10px; gap: 20px; flex-wrap: wrap;">
    <div style="text-align: center;">
        <img src="../../../img/first-principles-of-computer-vision/introduction-to-computer-vision-02.png" style="width: 300px;"alt="Black-and-white photo of two children showering — raw visual input"/>
        <p style="font-size: 13px; color: #888; margin-top: 4px;"><i>Raw visual input: two children showering</i></p>
    </div>
    <div style="text-align: center;">
        <img src="../../../img/first-principles-of-computer-vision/introduction-to-computer-vision-03.png" style="width: 320px;"alt="Numerical pixel matrix representation of the same photo"/>
        <p style="font-size: 13px; color: #888; margin-top: 4px;"><i>Same scene as a numerical pixel matrix</i></p>
    </div>
</div>

In this field, the definition of the mission often dictates the methodology. Below is a comparison of the three primary philosophical pillars of computer vision research:

| Perspective | Proponent | Core Philosophy |
|-------------|-----------|----------------|
| Vision as Emulation | David Marr | Aimed at automating human visual processes to replicate the sophistication of biological systems |
| Vision as Information Processing | Berthold Horn | Defined as the task of "inverting" image formation—mathematically walking back from a 2D projection to 3D reality |
| Vision as a Functional Tool | Takeo Kanade | Emphasizes that vision is "fun" but, more importantly, "useful," serving as a bridge between pure research and practical application |

### 1.1 The "First Principles" Philosophy

While contemporary deep learning provides powerful tools, a **First Principles** approach—focusing on mathematical and physical underpinnings—is the prerequisite for generalizable and explainable AI. Relying on "black box" models often bypasses the structural understanding required for true innovation.

> **Why First Principles?** Physical phenomena can be described with elegant math, making massive datasets and exhaustive training cycles unnecessary.

We prioritize these fundamentals for four reasons:

1. **Precision and Conciseness** — Physical phenomena can often be described with elegant math, rendering massive datasets unnecessary.
2. **Debugging and Diagnostics** — When a vision system fails, first principles provide the only rigorous framework for diagnosing the failure.
3. **Synthetic Data Generation** — When real-world data collection is impractical or dangerous, mathematical models allow us to synthesize high-fidelity training data.
4. **Scientific Curiosity** — The innate human drive to understand the "why" behind visual phenomena leads to breakthroughs that purely data-driven methods overlook.

---

## 2. The Human Visual System: Biology, Fallibility, and Ambiguity

Studying the human eye is a necessary starting point for designing artificial vision. Even though machines and humans often have divergent goals—qualitative navigation versus quantitative measurement—the eye provides a blueprint for efficient information reduction.

### 2.1 The Biological Pathway

The human visual system is a complex hierarchy designed for rapid analysis:

```mermaid
flowchart LR
    A["👁️ Eye & Lens<br/><i>Primary optical stage</i>"] --> B["🧬 Retina<br/><i>Early processing + data reduction</i>"]
    B --> C["🔌 Optic Nerve<br/><i>High-speed conduit</i>"]
    C --> D["🧠 LGN<br/><i>Relay station, directs to regions</i>"]
    D --> E["🎯 Visual Cortex<br/><i>Shape, color, motion, texture</i>"]
    
    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#16213e,stroke:#e94560,color:#fff
    style C fill:#0f3460,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#16213e,stroke:#e94560,color:#fff
```

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px;">
    <img src="../../../img/first-principles-of-computer-vision/introduction-to-computer-vision-04.png" style="display:flex; justify-content: center; width: 100%; max-width: 900px;"alt="Detailed brain anatomy: eye signals through LGN to visual cortex (V1, V2, MT/V5, V8)"/>
</div>
<p style="text-align: center; font-size: 14px; color: #888; margin-top: -10px;"><i>Biological vision pathway: retina → LGN → visual cortex (V1, V2, MT/V5, V8)</i></p>

### 2.2 Qualitative vs. Quantitative Vision

As engineers, we must recognize that **human vision is a qualitative system**, whereas factory automation, medical imaging, and robotics require quantitative precision. A human can recognize a face instantly but cannot measure the length of a component to the nearest millimeter. For tasks requiring extreme reliability, emulating human biology is often the "wrong" goal; machines must provide the measurable accuracy that biological systems lack.

### 2.3 Case Studies in Visual Illusions

Human vision is more fallible than it appears, often relying on internal assumptions to resolve ambiguity.

<br/>

**Example — Dongary Wave Illusion:** The static leaf pattern below appears to shimmer or move due to involuntary micro-saccades of the eye.

<div style="text-align: center;margin-bottom: 20px;">
    <img src="../../../img/first-principles-of-computer-vision/introduction-to-computer-vision-05.png" style="width: 400px;"alt="Leaf illusion — static leaves appearing to move due to involuntary eye movements"/>
    <p style="font-size: 13px; color: #888; margin-top: 4px;"><i>Static leaf pattern that produces a perception of motion — the Dongary Wave illusion</i></p>
</div>

| Illusion | What It Reveals |
|----------|-----------------|
| **Fraser's Spiral** | Concentric circles misinterpreted as a spiral |
| **Adelson's Checker Shadow** | Brain compensates for illumination — two identical gray patches look different |
| **Dongary Wave** | Perceived motion from a static image due to involuntary eye movements |
| **Ames Room** | Perspective and relative size create an illusion where people appear to grow or shrink |
| **Necker Cube / Faces vs. Vase** | A single 2D image supports multiple 3D or symbolic interpretations |
| **The Crater Illusion** | The "lighting from above" assumption — flipping a mound makes it appear as a crater |
| **Kanizsa Triangle** | The brain "fills in" data (thinking) rather than just processing pixels (seeing) |

> **Key Insight:** While humans *think* through their visual experiences, machines must first learn to *calculate* through the rigorous lens of radiometry and geometry.

---

## 3. Topics Covered: Roadmap

This specialization covers the entire pipeline — from pixels to perception — organized into six modules:

```mermaid
flowchart TB
    subgraph Foundations["🟦 Foundations"]
        direction TB
        A["Introduction<br/><i>What is CV, human vision</i>"]
        B["Imaging<br/><i>Formation, sensors, processing</i>"]
    end
    
    subgraph Features["🟧 Features & 2D"]
        C["Features<br/><i>Edges, SIFT, stitching, faces</i>"]
    end
    
    subgraph Reconstruction["🟩 3D Reconstruction"]
        D["Reconstruction I<br/><i>Radiometry, photometric stereo</i>"]
        E["Reconstruction II<br/><i>Stereo, optical flow, SfM</i>"]
    end
    
    subgraph Perception["🟥 Perception"]
        F["Perception<br/><i>Tracking, segmentation, NN</i>"]
    end
    
    A --> B --> C --> D --> E --> F
    
    style A fill:#1a1a2e,stroke:#4cc9f0,color:#fff
    style B fill:#1a1a2e,stroke:#4cc9f0,color:#fff
    style C fill:#1a1a2e,stroke:#f72585,color:#fff
    style D fill:#1a1a2e,stroke:#06d6a0,color:#fff
    style E fill:#1a1a2e,stroke:#06d6a0,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style Foundations fill:transparent,stroke:#4cc9f0,color:#4cc9f0
    style Features fill:transparent,stroke:#f72585,color:#f72585
    style Reconstruction fill:transparent,stroke:#06d6a0,color:#06d6a0
    style Perception fill:transparent,stroke:#e94560,color:#e94560
```

### Module Breakdown

| # | Module | Focus |
|---|--------|-------|
| 1 | **Imaging** | Image formation, sensors, binary images, image processing (convolution, Fourier) |
| 2 | **Features** | Edge/boundary detection, SIFT, image stitching, face detection |
| 3 | **Reconstruction I** | Radiometry, photometric stereo, shape from shading, depth from defocus |
| 4 | **Reconstruction II** | Camera calibration, stereo, optical flow, structure from motion |
| 5 | **Perception** | Object tracking, segmentation, appearance matching, neural networks |

---

## 4. Global Applications: Computer Vision in the Modern World

Vision has evolved from a laboratory curiosity into a thriving global industry across diverse sectors.

<br/>

| Domain | Applications |
|--------|-------------|
| **Industrial / Efficiency** | Factory automation, high-speed visual inspection, OCR for license plates and postal scanning |
| **Security / Identity** | Biometrics using iris patterns (as unique as DNA), robust face recognition |
| **Consumer Tech** | Optical mice (mini-vision systems), gaming (Kinect / PlayStation), AR (Snapchat 3D mesh filters) |
| **Intelligent Marketing** | Vending machines detecting customer demographics (age/gender) to display targeted products |
| **Visual Search** | Instant identification of monuments and objects via mobile devices |
| **Advanced Mobility** | Driverless cars using sensor fusion, Mars Rover terrain mapping |
| **Creative / Medical** | Motion capture for cinema, medical diagnostics (X-ray, MRI, ultrasound) |

---