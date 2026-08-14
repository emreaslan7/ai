# Phase Shifting Method, Structured Light Systems, and Time of Flight Method

<!-- toc -->

While discrete binary patterns allow unambiguous triangulation, achieving sub-pixel 3D accuracy requires projecting continuous intensity functions across the scene. In this chapter, we explore continuous phase shifting, high-profile industrial structured light applications, fundamental optical limits, and Time-of-Flight (ToF) range sensing.

---

## 1. Phase Shifting Method

Rather than projecting discrete binary stripes, the phase shifting method projects mathematical light patterns whose intensities vary continuously across space. This increases spatial resolution to sub-pixel accuracy.

### 1.1 Intensity Ratio Method

- **Ramp Function:** A single ramp illumination pattern $L_1$ is projected onto the scene, where intensity decreases linearly from maximum brightness at one side to zero at the other ($x_p$).
- **Flat Uniform Illumination:** A second image is captured under flat uniform light $L_2$.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/phase-shifting-structured-light-and-time-of-flight-01.png" alt="Intensity Ratio Method Projection Patterns" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 1: Projection of linear ramp pattern L1 and flat uniform pattern L2 [Carrihill 1985].</em></figcaption>
  </div>
</figure>

- **Normalization:** Measuring pixel intensities $I_1 = \rho \cdot L_1$ and $I_2 = \rho \cdot L_2$ in the camera and taking their ratio cancels the unknown surface albedo and surface normal factor $\rho$:

$$\frac{I_1}{I_2} = \frac{\rho \cdot L_1}{\rho \cdot L_2} = \frac{L_1}{L_2}$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/phase-shifting-structured-light-and-time-of-flight-02.png" alt="Intensity Ratio Normalization" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 2: Ratioing I1/I2 eliminates albedo variation, yielding direct mapping to projector column coordinate x_p.</em></figcaption>
  </div>
</figure>

- **Disadvantage:** Highly sensitive to sensor noise and projector intensity quantization steps.

### 1.2 Sinusoidal Phase Shifting Mathematics

In industrial automation and quality inspection, the gold-standard technique is **Sinusoidal Phase Shifting**, which projects continuous cosine waves onto the scene and shifts their phase temporally.

The emitted projector cosine wave is defined by average brightness $b$, amplitude $b$, and period $P$. Accounting for unknown ambient lighting $a$ and relative surface albedo $\rho$, the pixel intensity observed by the camera is:

$$I_1(x_c, y_c) = \rho a + \rho b + \rho b \cos\left( \frac{2\pi x_p}{P} \right)$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/phase-shifting-structured-light-and-time-of-flight-03.png" alt="Sinusoidal Cosine Wave Projection" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 3: First reference cosine wave L1 projected onto the scene [Wust 1991].</em></figcaption>
  </div>
</figure>

This equation contains three unknowns: $\rho a$ (ambient component), $\rho b$ (amplitude component), and the target projector column coordinate $x_p$. To solve for these three unknowns, exactly three phase-shifted images are captured:

1. **Frame 1 ($I_1$):** Reference cosine pattern $L_1$ projected with $0^\circ$ phase shift.
2. **Frame 2 ($I_2$):** Pattern phase shifted by $-120^\circ$ ($-2\pi/3$).

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/phase-shifting-structured-light-and-time-of-flight-04.png" alt="Phase Shift -120 degrees" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 4: Second cosine pattern L2 shifted by -120° (-2π/3).</em></figcaption>
  </div>
</figure>

3. **Frame 3 ($I_3$):** Pattern phase shifted by $+120^\circ$ ($+2\pi/3$).

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/phase-shifting-structured-light-and-time-of-flight-05.png" alt="Phase Shift +120 degrees" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 5: Third cosine pattern L3 shifted by +120° (+2π/3).</em></figcaption>
  </div>
</figure>

Solving these three simultaneous trigonometric equations eliminates ambient lighting $\rho a$ and amplitude $\rho b$, yielding a closed-form solution for projector column $x_p$:

$$x_p = \frac{P}{2\pi} \tan^{-1}\left( \sqrt{3} \frac{I_2 - I_3}{2I_1 - I_2 - I_3} \right)$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/phase-shifting-structured-light-and-time-of-flight-06.png" alt="Phase Shifting Closed-Form Solution" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 6: Closed-form trigonometric solution for projector column coordinate x_p using 3 phase-shifted images.</em></figcaption>
  </div>
</figure>

Intersecting the computed projector column plane $x_p$ with the camera viewing ray yields sub-millimeter 3D point accuracy.

---

## 2. Structured Light Systems

### 2.1 Notable High-Profile Systems

- **3D Visual Inspection (Omron Corp.):** Used in surface-mount factory assembly lines to inspect printed circuit board (PCB) solder joints and micro-components in real time. The PCB is tiled and scanned via phase shifting in seconds to reject defective solder joints instantly.
- **Digital Michelangelo Project (Levoy 2000):** Stanford researchers scanned Michelangelo's *David* statue in Florence over 30 nights using precision structured light range scanners. Achieving a mesh resolution of $1/4 \text{ mm}$, the project created a permanent digital twin (*Virtual David*) for micro-erosion tracking and archival preservation.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/phase-shifting-structured-light-and-time-of-flight-08.png" alt="Digital Michelangelo Project Scanning David Statue" style="display:flex; border-radius: 5px; justify-content: center; width: 520px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 7: Digital Michelangelo Project: High-resolution 3D mesh (1/4 mm accuracy) of Michelangelo's David statue [Levoy 2000].</em></figcaption>
  </div>
</figure>

- **Great Buddha Project (Ikeuchi 2007):** Drone-mounted structured light and laser scanners were deployed in Nara, Japan to digitize the monumental Great Buddha statue and surrounding temple heritage structures.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/phase-shifting-structured-light-and-time-of-flight-09.png" alt="Great Buddha Project" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 8: Great Buddha Project: Physical statue in Nara and its 3D digital model [Ikeuchi 2007].</em></figcaption>
  </div>
</figure>

### 2.2 Limitations and Unsolved Problems

Despite high accuracy, structured light systems face physical limitations on certain surface and material types:

1. **Specular / Metallic Surfaces:** Mirror-like specular reflection redirects light exclusively along the angle of reflection. Light rarely backscatters to the camera, leaving empty holes in the depth map.
2. **Translucent / Subsurface Scattering Surfaces:** On materials like marble, wax, or human skin, light penetrates beneath the surface and scatters internally before exiting from adjacent pixels, destroying pattern edge sharpness.
3. **Participating Media:** In fog, smoke, or turbid underwater environments, light attenuates rapidly and ambient scattering causes the medium itself to glow, masking projected patterns.
4. **Transparent Objects (Glass/Water):** Light refracts directly through glass objects without scattering.
5. **Hair and Micro-Fibers:** Hair strands are far smaller than an individual camera pixel, causing multiple strands to project onto a single pixel and breaking geometric triangulation.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/phase-shifting-structured-light-and-time-of-flight-10.png" alt="Unsolved Problems in Structured Light" style="display:flex; border-radius: 5px; justify-content: center; width: 520px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 9: Challenging surfaces for structured light: Subsurface scattering (marble), participating media (underwater), specular metal, transparent glass, and hair.</em></figcaption>
  </div>
</figure>

### 2.3 Summary of Structured Light Methods

The table below summarizes the image count complexity of all major structured light range finding paradigms:

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/phase-shifting-structured-light-and-time-of-flight-07.png" alt="Structured Light Methods Summary Table" style="display:flex; border-radius: 5px; justify-content: center; width: 480px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 10: Summary table comparing required image frame counts across structured light paradigms.</em></figcaption>
  </div>
</figure>

---

## 3. Time of Flight Method (ToF)

Time of Flight (ToF) range sensing bypasses baseline triangulation entirely by directly measuring the round-trip travel time of light ($c \approx 3 \times 10^8 \text{ m/s}$).

### 3.1 Biological Origins and Historical Speed of Light Experiments

- **Biological Biosonar:** Bats, dolphins, and whales use echolocation (sonar) by emitting sound waves and timing returning echoes to perceive 3D space. ToF applies this exact principle using light.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/phase-shifting-structured-light-and-time-of-flight-11.png" alt="Echolocation in Nature" style="display:flex; border-radius: 5px; justify-content: center; width: 420px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 11: Biological origins of Time-of-Flight: Echolocation using sound waves in bats, dolphins, and submarines.</em></figcaption>
  </div>
</figure>

- **Galileo's Lantern Experiment (1600s):** Galileo attempted to measure light speed by placing two lantern operators on hilltops 1000 meters apart (2000m round trip). Since light travels 2000m in just $6.6 \ \mu\text{s}$, human muscle reflexes (~milliseconds) rendered the experiment unsuccessful.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/phase-shifting-structured-light-and-time-of-flight-12.png" alt="Galileo's Speed of Light Experiment" style="display:flex; border-radius: 5px; justify-content: center; width: 480px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 12: Galileo's early attempt to measure the speed of light across 1000m hilltop baselines.</em></figcaption>
  </div>
</figure>

- **Fizeau's Cogwheel Experiment (1849):** Hippolyte Fizeau successfully measured light speed by passing light through a rapidly spinning cogwheel over an $8633 \text{ m}$ distance to a plane mirror. By measuring the rotational speed at which returning light was blocked by adjacent teeth, he calculated $c_{\text{computed}} \approx 3.153 \times 10^8 \text{ m/s}$ (remarkably close to actual $2.998 \times 10^8 \text{ m/s}$).

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/phase-shifting-structured-light-and-time-of-flight-13.png" alt="Fizeau's Cogwheel Experiment" style="display:flex; border-radius: 5px; justify-content: center; width: 480px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 13: Fizeau's 1849 cogwheel setup for measuring the speed of light over an 8633m baseline.</em></figcaption>
  </div>
</figure>

### 3.2 Pulse Modulation (Flash ToF)

- **Operating Principle:** A short, high-power laser pulse is emitted into the scene. An ultra-fast nanosecond stopwatch measures the time delay $\Delta t$ before the reflected pulse strikes the sensor.
- **Disadvantage:** Sub-centimeter precision requires sub-nanosecond stopwatch electronics and high peak-power pulsed lasers, making high-resolution arrays costly.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/phase-shifting-structured-light-and-time-of-flight-14.png" alt="Pulse Modulation ToF" style="display:flex; border-radius: 5px; justify-content: center; width: 450px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 14: Pulse modulation (Flash ToF): Measuring exact pulse time delay using nanosecond timing circuits.</em></figcaption>
  </div>
</figure>

### 3.3 Continuous Modulation (Phase ToF)

To avoid sub-nanosecond digital stopwatches, continuous modulation modulates emitted light intensity continuously using a high-frequency sinusoid (e.g., $f = 30 \text{ MHz}$).

Depth is directly proportional to the phase shift $\varphi$ measured between the emitted and returning cosine waves.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/phase-shifting-structured-light-and-time-of-flight-15.png" alt="Continuous Modulation Phase ToF" style="display:flex; border-radius: 5px; justify-content: center; width: 450px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 15: Continuous modulation ToF: Phase shift φ between emitted and received sinusoidal light signals.</em></figcaption>
  </div>
</figure>

#### Correlation-Based Phase Measurement

The returning optical signal is demodulated by multiplying and integrating pixel charge against a reference signal $S_{ref}$ phase-locked to the emitter:

$$L_{emit} = \cos(\omega t)$$

$$L_{scene} = O + A \cos(\omega t - \varphi)$$

$$S_{ref} = \cos(\omega t - \delta)$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/phase-shifting-structured-light-and-time-of-flight-16.png" alt="Correlation-Based Phase Measurement Setup" style="display:flex; border-radius: 5px; justify-content: center; width: 480px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 16: Correlation-based phase measurement parameters: Ambient light O, albedo A, phase shift φ, and reference phase δ.</em></figcaption>
  </div>
</figure>

Measuring pixel charge under three distinct reference phase shifts ($\delta_1, \delta_2, \delta_3$) allows closed-form recovery of the unknown phase delay $\varphi$.

#### Phase-to-Distance Formula

Once phase shift $\varphi$ is recovered, absolute distance $d$ is given by:

$$d = c \frac{\varphi}{4\pi f}$$

> **Numerical Example:** For modulation frequency $f = 30 \text{ MHz}$ and detected phase shift $\varphi = \pi$:
> $$d = (3 \times 10^8) \cdot \frac{\pi}{4\pi \cdot (30 \times 10^6)} = \frac{3 \times 10^8}{1.2 \times 10^8} = 2.5 \text{ meters}$$

### 3.4 Industrial Applications and Mobile Devices

- **Autonomous Vehicles (LiDAR):** Mechanical rotating LiDAR systems sweep single laser beams across $360^\circ$ to generate dense 3D point clouds for autonomous navigation. Solid-state LiDAR architectures are rapidly reducing cost and size.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../img/first-principles-of-computer-vision/phase-shifting-structured-light-and-time-of-flight-17.png" alt="Google Self-Driving Car 3D Point Cloud" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Figure 17: Autonomous vehicle 3D point cloud generation using scanning LiDAR / Time-of-Flight sensors.</em></figcaption>
  </div>
</figure>

- **Mobile Consumer Devices (Solid-State ToF):** Modern smartphones and tablets integrate solid-state ToF sensor arrays. Instead of mechanical scanning, every pixel measures phase shift simultaneously, generating real-time depth maps for Augmented Reality (AR), portrait bokeh, and facial recognition.
