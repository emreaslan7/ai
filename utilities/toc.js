const currentURL = window.location.href;

function getThemeColors(theme) {
  const themes = {
    ayu: {
      baseColor: "#c5c5c5",
      activeColor: "#ffb454",
      hoverColor: "#b7b9cc",
    },
    coal: {
      baseColor: "#98a3ad",
      activeColor: "#3473ad",
      hoverColor: "#b3c0cc",
    },
    light: {
      baseColor: "#88848a",
      activeColor: "#000",
      hoverColor: "hsl(0, 4.70%, 74.90%)",
    },
    navy: {
      baseColor: "#bcbdd0",
      activeColor: "#2b79a2",
      hoverColor: "#b7b9cc",
    },
    rust: {
      baseColor: "#bdbdbd",
      activeColor: "#e69f67",
      hoverColor: "#e8aa2e",
    },
  };
  return themes[theme] || themes.light;
}

function toggleVisibility(id) {
  const element = document.getElementById(id);
  if (!element) return;

  const iconSpan = document.getElementById(id + "-icon");
  const title = element.previousElementSibling;

  if (element.style.display === "none") {
    element.style.display = "block";
    if (iconSpan) {
      iconSpan.textContent = "▼";
    } else if (title) {
      title.innerHTML = `▼ ${title.textContent.trim().replace(/^▶|^▼/, "")}`;
    }
    localStorage.setItem(id, "open"); // Durumu kaydet
  } else {
    element.style.display = "none";
    if (iconSpan) {
      iconSpan.textContent = "▶";
    } else if (title) {
      title.innerHTML = `▶ ${title.textContent.trim().replace(/^▶|^▼/, "")}`;
    }
    localStorage.setItem(id, "closed"); // Durumu kaydet
  }
}

function HeadingCollapsible(text, id, fontSize = "15px", fontWeight = "bold") {
  const isOpen = localStorage.getItem(id) === "open"; // Önceki durumu kontrol et
  const displayStyle = isOpen ? "block" : "none";
  const icon = isOpen ? "▼" : "▶";

  return `
    <p 
      style="font-size: ${fontSize}; font-weight: ${fontWeight}; cursor: pointer; user-select: none; padding: 5px 0;" 
      onclick="toggleVisibility('${id}')"
    >
      ${icon} ${text}
    </p>
    <div id="${id}" style="display: ${displayStyle}; padding-left: 15px;">
  `;
}

function SubHeadingCollapsible(
  id,
  number,
  href,
  text,
  theme,
  fontSize = "13px",
  fontWeight = "bold",
  sublist = ""
) {
  const isOpen = localStorage.getItem(id) === "open";
  const displayStyle = isOpen ? "block" : "none";
  const icon = isOpen ? "▼" : "▶";
  const headingNumberSpan = createHeadingNumberSpan(number);
  const link = href ? createLink(href, text, theme) : text;

  return `<li style="margin: 7px 0px; font-size: ${fontSize}; list-style-type: none; padding-left: 0; font-weight: ${fontWeight};"><span style="cursor: pointer; user-select: none;" onclick="toggleVisibility('${id}')"><span id="${id}-icon">${icon}</span> ${headingNumberSpan}${link}</span><div id="${id}" style="display: ${displayStyle};">${sublist}</div></li>`;
}


function createHeadingNumberSpan(number) {
  if (!number) return "";
  return `<span style="font-weight: bold; margin-right: 5px;">${number}</span>`;
}

function createLink(href, text, theme) {
  const currentURL = window.location.pathname;
  const isActive = currentURL === href;

  const { baseColor, activeColor, hoverColor } = getThemeColors(theme);

  const baseStyle = `text-decoration: none; color: ${baseColor}; margin-right: 5px;`;
  const activeStyle = `font-weight: bold; color: ${activeColor};`;
  const hoverStyle = `color: ${hoverColor};`;

  return `<a href="${href}" style="${baseStyle} ${isActive ? activeStyle : ""
    }" onmouseover="this.style.color='${hoverColor}'" onmouseout="this.style.color='${isActive ? activeColor : baseColor
    }'">${text}</a>`;
}

function SubHeading(
  number,
  href,
  text,
  theme,
  fontSize = "12px",
  fontWeight = "normal",
  sublist = ""
) {
  const headingNumberSpan = createHeadingNumberSpan(number);
  const link = href ? createLink(href, text, theme) : text; // Eğer href yoksa link oluşturma

  return `<li style="margin: 7px 0px; font-size: ${fontSize}; list-style-type: none; padding-left: 0; font-weight: ${fontWeight};">${headingNumberSpan}${link}${sublist}</li>`;
}

function SubHeadingList(items, indentation = false) {
  const indentationCSS = indentation
    ? "padding-left: 20px;"
    : "padding-left: 0;";
  return `<ul style="list-style-type: none; ${indentationCSS} ">${items.join(
    ""
  )}</ul>`;
}

function Heading(text, fontSize = "14px", fontWeight = "bold") {
  return `<p style="font-size: ${fontSize}; font-weight: ${fontWeight};">${text}</p>`;
}

function updateTOC(url, theme) {
  const tocElement = document.querySelector(
    "#mdbook-sidebar .sidebar-scrollbox"
  );

  if (!tocElement) return;

  const tocContentEn = `
    ${SubHeading("", "/", "Welcome", theme, "13px", "bold")}
    ${HeadingCollapsible(
    "Machine Learning Specialization",
    "ml-specialization"
  )} 
    ${SubHeading("", "/machine-learning-specialization", "Content", theme)}
    ${SubHeadingList([
    SubHeading(
      "1.",
      "",
      "Supervised Machine Learning: Regression and Classification",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "1.1",
            "/machine-learning-specialization/supervised-machine-learning-regression-and-classification/supervised-unsupervised-learning.html",
            "Supervised and Unsupervised Learning",
            theme
          ),
          SubHeading(
            "1.2",
            "/machine-learning-specialization/supervised-machine-learning-regression-and-classification/linear-regression-and-cost-function.html",
            "Linear Regression and Cost Function",
            theme
          ),
          SubHeading(
            "1.3",
            "/machine-learning-specialization/supervised-machine-learning-regression-and-classification/gradient-descent.html",
            "Gradient Descent",
            theme
          ),
          SubHeading(
            "1.4",
            "/machine-learning-specialization/supervised-machine-learning-regression-and-classification/multiple-features.html",
            "Multiple Features",
            theme
          ),
          SubHeading(
            "1.5",
            "/machine-learning-specialization/supervised-machine-learning-regression-and-classification/feature-engineering-and-polynomial-regression.html",
            "Feature Engineering and Polynomial Regression",
            theme
          ),
          SubHeading(
            "1.6",
            "/machine-learning-specialization/supervised-machine-learning-regression-and-classification/classification-with-logistic-regression.html",
            "Classification with Logistic Regression",
            theme
          ),
          SubHeading(
            "1.7",
            "/machine-learning-specialization/supervised-machine-learning-regression-and-classification/overfitting-and-regularization.html",
            "Overfitting and Regularization",
            theme
          ),
          SubHeading(
            "1.8",
            "/machine-learning-specialization/supervised-machine-learning-regression-and-classification/scikit-learn-practical-applications.html",
            "Scikit-Learn: Practical Applications",
            theme
          ),
        ],
        true
      )
    ),
  ])}
    ${SubHeadingList([
    SubHeading(
      "2.",
      "",
      "Advanced Learning Algorithms",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "2.1",
            "/machine-learning-specialization/advanced-learning-algorithms/neural-networks-intuition-and-model.html",
            "Neural Networks: Intuition and Model",
            theme
          ),
          SubHeading(
            "2.2",
            "/machine-learning-specialization/advanced-learning-algorithms/implementation-of-forward-propagation.html",
            "Implementation of Forward Propagation",
            theme
          ),
          SubHeading(
            "2.3",
            "/machine-learning-specialization/advanced-learning-algorithms/neural-network-training-and-activation-functions.html",
            "Neural Network Training and Activation Functions",
            theme
          ),
          SubHeading(
            "2.4",
            "/machine-learning-specialization/advanced-learning-algorithms/optimizers-and-layer-types.html",
            "Optimizers and Layer Types",
            theme
          ),
          SubHeading(
            "2.5",
            "/machine-learning-specialization/advanced-learning-algorithms/model-evaluation-selection-and-improvement.html",
            "Model Evaluation, Selection, and Improvement",
            theme
          ),
          SubHeading(
            "2.6",
            "/machine-learning-specialization/advanced-learning-algorithms/decision-trees.html",
            "Decision Trees",
            theme
          ),
        ],
        true
      )
    ),
    SubHeading(
      "3.",
      "",
      "Unsupervised Learning, Recommenders, Reinforcement Learning",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "3.1",
            "",
            "Unsupervised Learning",
            theme,
            "13px",
            "normal",
            SubHeadingList(
              [
                SubHeading(
                  "3.1.1",
                  "/machine-learning-specialization/unsupervised-learning-recommenders-reinforcement-learning/k-means-clustering.html",
                  "K-Means Clustering",
                  theme
                ),
                SubHeading(
                  "3.1.2",
                  "/machine-learning-specialization/unsupervised-learning-recommenders-reinforcement-learning/anomaly-detection.html",
                  "Anomaly Detection",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "3.2",
            "/machine-learning-specialization/unsupervised-learning-recommenders-reinforcement-learning/recommender-systems.html",
            "Recommender Systems",
            theme
          ),
          SubHeading(
            "3.3",
            "/machine-learning-specialization/unsupervised-learning-recommenders-reinforcement-learning/reinforcement-learning.html",
            "Reinforcement Learning",
            theme
          ),
        ],
        true
      )
    ),
  ])}
    </div>

    ${HeadingCollapsible("Deep Learning Specialization", "dl-specialization")} 
    ${SubHeading("", "/deep-learning-specialization", "Content", theme)}
      ${SubHeadingList([
    SubHeading(
      "1.",
      "",
      "Convolutional Neural Networks",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "1.1",
            "",
            "Foundations of Convolutional Neural Networks",
            theme,
            "11px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "1.1.1",
                  "/deep-learning-specialization/convolutional-neural-networks/computer-vision-and-edge-detection.html",
                  "Computer Vision and Edge Detection",
                  theme
                ),
                SubHeading(
                  "1.1.2",
                  "/deep-learning-specialization/convolutional-neural-networks/convolutional-operations.html",
                  "Convolutional Operations",
                  theme
                ),
                SubHeading(
                  "1.1.3",
                  "/deep-learning-specialization/convolutional-neural-networks/cnn-architecture-and-examples.html",
                  "CNN Architecture and Examples",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "1.2",
            "",
            "Deep Convolutional Models: Cases Studies",
            theme,
            "11px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "1.2.1",
                  "/deep-learning-specialization/convolutional-neural-networks/classic-networks-lenet-alexnet-vgg.html",
                  "Classic Networks: LeNet-5, AlexNet, VGG",
                  theme
                ),
                SubHeading(
                  "1.2.2",
                  "/deep-learning-specialization/convolutional-neural-networks/modern-cnn-architectures-resnet-inception-mobilenet-efficenet.html",
                  "Modern CNN Architectures: ResNet, Inception, MobileNet, EfficientNet",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "1.3",
            "",
            "Object Detection and Face Recognition",
            theme,
            "11px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "1.3.1",
                  "/deep-learning-specialization/convolutional-neural-networks/object-localization-and-detection.html",
                  "Object Localization and Detection",
                  theme
                ),
                SubHeading(
                  "1.3.2",
                  "/deep-learning-specialization/convolutional-neural-networks/evaluation-and-optimization-iou-nms-anchor-boxes.html",
                  "Evaluation and Optimization: IoU, Non-max Suppression, Anchor Boxes",
                  theme
                ),
                SubHeading(
                  "1.3.3",
                  "/deep-learning-specialization/convolutional-neural-networks/region-proposals-and-semantic-segmentation-unet.html",
                  "Region Proposals and Semantic Segmentation: U-Net",
                  theme
                ),
                SubHeading(
                  "1.3.4",
                  "/deep-learning-specialization/convolutional-neural-networks/face-recognition-and-neural-style-transfer.html",
                  "Face Recognition and Neural Style Transfer",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "1.4",
            "",
            "Sequence Models",
            theme,
            "11px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "1.4.1",
                  "/deep-learning-specialization/sequence-models/recurrent-neural-networks.html",
                  "Recurrent Neural Networks",
                  theme
                ),
                SubHeading(
                  "1.4.2",
                  "/deep-learning-specialization/sequence-models/natural-language-processing-and-word-embeddings.html",
                  "Natural Language Processing and Word Embeddings",
                  theme
                ),
              ],
              true
            )
          ),
        ],
        true
      )
    ),
  ])}
    </div>

    ${HeadingCollapsible("First Principles of Computer Vision", "fpcv-specialization")} 
    ${SubHeading("", "/first-principles-of-computer-vision", "Content", theme)}
      ${SubHeadingList([
    SubHeadingCollapsible(
      "fpcv-1",
      "1.",
      "",
      "Introduction",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "1.1",
            "/first-principles-of-computer-vision/introduction-to-computer-vision.html",
            "Introduction to Computer Vision",
            theme
          ),
        ],
        true
      )
    ),
    SubHeadingCollapsible(
      "fpcv-2",
      "2.",
      "",
      "Imaging",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "2.1",
            "",
            "Image Formation",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "2.1.1",
                  "/first-principles-of-computer-vision/imaging/image-formation/pinhole-and-perspective-projection.html",
                  "Pinhole Camera Model and Perspective Projection",
                  theme
                ),
                SubHeading(
                  "2.1.2",
                  "/first-principles-of-computer-vision/imaging/image-formation/lenses-and-depth-of-field.html",
                  "Lens Systems and Depth of Field",
                  theme
                ),
                SubHeading(
                  "2.1.3",
                  "/first-principles-of-computer-vision/imaging/image-formation/advanced-optical-systems.html",
                  "Advanced Optical Systems",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "2.2",
            "",
            "Image Sensing",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "2.2.1",
                  "/first-principles-of-computer-vision/imaging/image-sensing/overview-history-and-sensor-types.html",
                  "Overview, History, and Image Sensor Types",
                  theme
                ),
                SubHeading(
                  "2.2.2",
                  "/first-principles-of-computer-vision/imaging/image-sensing/resolution-noise-and-color-sensing.html",
                  "Resolution, Noise, Dynamic Range, and Color Sensing",
                  theme
                ),
                SubHeading(
                  "2.2.3",
                  "/first-principles-of-computer-vision/imaging/image-sensing/camera-response-hdr-and-nature-sensors.html",
                  "Camera Response, HDR Imaging, and Nature's Sensors",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "2.3",
            "",
            "Binary Images",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "2.3.1",
                  "/first-principles-of-computer-vision/imaging/binary-images/overview-and-geometric-properties.html",
                  "Overview and Geometric Properties",
                  theme
                ),
                SubHeading(
                  "2.3.2",
                  "/first-principles-of-computer-vision/imaging/binary-images/segmenting-binary-images-and-iterative-modification.html",
                  "Segmenting Binary Images and Iterative Modification",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "2.4",
            "",
            "Image Processing I",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "2.4.1",
                  "/first-principles-of-computer-vision/imaging/image-processing-1/pixel-processing-lsis-and-convolution.html",
                  "Pixel Processing, LSIS, and Continuous Convolution",
                  theme
                ),
                SubHeading(
                  "2.4.2",
                  "/first-principles-of-computer-vision/imaging/image-processing-1/linear-and-non-linear-filters.html",
                  "Linear and Non-Linear Image Filters",
                  theme
                ),
                SubHeading(
                  "2.4.3",
                  "/first-principles-of-computer-vision/imaging/image-processing-1/template-matching.html",
                  "Template Matching",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "2.5",
            "",
            "Image Processing II",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "2.5.1",
                  "/first-principles-of-computer-vision/imaging/image-processing-2/overview-fourier-transform-and-convolution-theorem.html",
                  "Overview, Fourier Transform, and Convolution Theorem",
                  theme
                ),
                SubHeading(
                  "2.5.2",
                  "/first-principles-of-computer-vision/imaging/image-processing-2/filtering-in-frequency-domain-and-deconvolution.html",
                  "Filtering in Frequency Domain and Deconvolution",
                  theme
                ),
                SubHeading(
                  "2.5.3",
                  "/first-principles-of-computer-vision/imaging/image-processing-2/sampling-theory-and-aliasing.html",
                  "Sampling Theory and Aliasing",
                  theme
                ),
              ],
              true
            )
          ),
        ],
        true
      )
    ),
    SubHeadingCollapsible(
      "fpcv-3",
      "3.",
      "",
      "Features and Boundaries",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "3.1",
            "",
            "Edge Detection",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "3.1.1",
                  "/first-principles-of-computer-vision/features/edge-detection/overview-gradients-and-laplacian.html",
                  "Overview, Gradients, and Laplacian Edge Detection",
                  theme
                ),
                SubHeading(
                  "3.1.2",
                  "/first-principles-of-computer-vision/features/edge-detection/canny-and-corner-detection.html",
                  "Canny Edge Detector and Corner Detection",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "3.2",
            "",
            "Boundary Detection",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "3.2.1",
                  "/first-principles-of-computer-vision/features/boundary-detection/overview-fitting-lines-and-active-contours.html",
                  "Overview, Fitting Lines and Curves, and Active Contours",
                  theme
                ),
                SubHeading(
                  "3.2.2",
                  "/first-principles-of-computer-vision/features/boundary-detection/hough-transform-and-generalized-hough-transform.html",
                  "Hough Transform and Generalized Hough Transform",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "3.3",
            "",
            "SIFT Detector",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "3.3.1",
                  "/first-principles-of-computer-vision/features/sift-detector/sift-detector.html",
                  "SIFT Detector and Descriptor",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "3.4",
            "",
            "Image Stitching",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "3.4.1",
                  "/first-principles-of-computer-vision/features/image-stitching/overview-and-image-transformations.html",
                  "Overview and Image Transformations",
                  theme
                ),
                SubHeading(
                  "3.4.2",
                  "/first-principles-of-computer-vision/features/image-stitching/homography-ransac-warping-and-blending.html",
                  "Homography, RANSAC, Warping and Blending",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "3.5",
            "",
            "Face Detection",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "3.5.1",
                  "/first-principles-of-computer-vision/features/face-detection/face-detection.html",
                  "Face Detection",
                  theme
                ),
              ],
              true
            )
          ),

        ],
        true
      )
    ),
    SubHeadingCollapsible(
      "fpcv-4",
      "4.",
      "",
      "3D Reconstruction - Single Viewpoint",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "4.1",
            "",
            "Radiometry and Reflectance",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "4.1.1",
                  "/first-principles-of-computer-vision/reconstruction-i/radiometry-and-reflectance/overview-radiometric-concepts-and-brdf.html",
                  "Overview, Radiometric Concepts, Radiance, and BRDF",
                  theme
                ),
                SubHeading(
                  "4.1.2",
                  "/first-principles-of-computer-vision/reconstruction-i/radiometry-and-reflectance/reflectance-models-rough-surfaces-and-dichromatic-model.html",
                  "Reflectance Models, Rough Surfaces, and Dichromatic Model",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "4.2",
            "",
            "Photometric Stereo",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "4.2.1",
                  "/first-principles-of-computer-vision/reconstruction-i/photometric-stereo/overview-gradient-space-and-lambertian-case.html",
                  "Overview, Gradient Space, Reflectance Map, and Lambertian Case",
                  theme
                ),
                SubHeading(
                  "4.2.2",
                  "/first-principles-of-computer-vision/reconstruction-i/photometric-stereo/calibration-shape-from-normals-and-interreflections.html",
                  "Calibration-Based Photometric Stereo, Shape from Normals, and Interreflections",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "4.3",
            "",
            "Shape from Shading",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "4.3.1",
                  "/first-principles-of-computer-vision/reconstruction-i/shape-from-shading/shape-from-shading.html",
                  "Overview, Human Perception, Stereographic Projection, SfS Algorithm, and Illusions",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "4.4",
            "",
            "Depth from Defocus",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "4.4.1",
                  "/first-principles-of-computer-vision/reconstruction-i/depth-from-defocus/depth-from-defocus.html",
                  "Depth from Focus & Defocus",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "4.5",
            "",
            "Active Illumination Methods",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "4.5.1",
                  "/first-principles-of-computer-vision/reconstruction-i/active-illumination/overview-photometric-stereo-and-structured-light-range-finding.html",
                  "Overview, Photometric Stereo Systems, and Structured Light Range Finding",
                  theme
                ),
                SubHeading(
                  "4.5.2",
                  "/first-principles-of-computer-vision/reconstruction-i/active-illumination/phase-shifting-structured-light-and-time-of-flight.html",
                  "Phase Shifting Method, Structured Light Systems, and Time of Flight Method",
                  theme
                ),
              ],
              true
            )
          ),
        ],
        true
      )
    ),
    SubHeadingCollapsible(
      "fpcv-5",
      "5.",
      "",
      "3D Reconstruction - Multiple Viewpoints",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "5.1",
            "",
            "Camera Calibration",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "5.1.1",
                  "/first-principles-of-computer-vision/reconstruction-ii/camera-calibration/camera-models-and-calibration.html",
                  "Camera Models and Calibration",
                  theme
                ),
                SubHeading(
                  "5.1.2",
                  "/first-principles-of-computer-vision/reconstruction-ii/camera-calibration/simple-stereo.html",
                  "Simple Stereo",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "5.2",
            "",
            "Uncalibrated Stereo",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "5.2.1",
                  "/first-principles-of-computer-vision/reconstruction-ii/camera-calibration/uncalibrated-stereo.html",
                  "Uncalibrated Stereo Vision and Epipolar Geometry",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "5.3",
            "",
            "Optical Flow",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "5.3.1",
                  "/first-principles-of-computer-vision/reconstruction-ii/camera-calibration/optical-flow.html",
                  "Optical Flow and Motion Analysis",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "5.4",
            "",
            "Structure from Motion",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "5.4.1",
                  "/first-principles-of-computer-vision/reconstruction-ii/camera-calibration/structure-from-motion.html",
                  "Structure from Motion and Tomasi-Kanade Factorization",
                  theme
                ),
              ],
              true
            )
          ),
        ],
        true
      )
    ),
    SubHeadingCollapsible(
      "fpcv-6",
      "6.",
      "",
      "Perception",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "6.1",
            "",
            "Object Tracking",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "6.1.1",
                  "/first-principles-of-computer-vision/perception/object-tracking/object-tracking.html",
                  "Object Tracking & Background Subtraction",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "6.2",
            "",
            "Image Segmentation",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "6.2.1",
                  "/first-principles-of-computer-vision/perception/image-segmentation/image-segmentation.html",
                  "Image Segmentation",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "6.4",
            "",
            "Appearance Matching",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "6.4.1",
                  "/first-principles-of-computer-vision/perception/appearance-matching/appearance-representation-and-pca.html",
                  "Appearance Representation and PCA",
                  theme
                ),
                SubHeading(
                  "6.4.2",
                  "/first-principles-of-computer-vision/perception/appearance-matching/svd-parametric-manifolds-and-appearance-matching.html",
                  "SVD, Parametric Manifolds, and Appearance Matching",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "6.5",
            "",
            "Neural Networks",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "6.5.1",
                  "/first-principles-of-computer-vision/perception/neural-networks/perceptron-and-activation-functions.html",
                  "Perceptron and Activation Functions",
                  theme
                ),
                SubHeading(
                  "6.5.2",
                  "/first-principles-of-computer-vision/perception/neural-networks/multilayer-neural-networks-and-backpropagation.html",
                  "Multilayer Networks, Gradient Descent, and Backpropagation",
                  theme
                ),
              ],
              true
            )
          ),
        ],
        true
      )
    ),
  ])}
    </div>

    ${HeadingCollapsible("Deep Learning with PyTorch", "dlwpt-project")} 
    ${SubHeading("", "/deep-learning-with-pytorch", "Content", theme)}
    ${SubHeadingList([
    SubHeadingCollapsible(
      "dlwpt-1",
      "1.",
      "",
      "Core PyTorch",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "1.1",
            "/deep-learning-with-pytorch/part-1-core-pytorch/introducing-deep-learning-and-the-pytorch-library.html",
            "Introducing Deep Learning and the PyTorch Library",
            theme
          ),
          SubHeading(
            "1.2",
            "/deep-learning-with-pytorch/part-1-core-pytorch/pretrained-networks-and-model-zoo.html",
            "Pretrained Networks and the Model Zoo",
            theme
          ),
          SubHeading(
            "1.3",
            "/deep-learning-with-pytorch/part-1-core-pytorch/it-starts-with-a-tensor.html",
            "It Starts with a Tensor: Storage, Strides, and Memory Layouts",
            theme
          ),
          SubHeading(
            "1.4",
            "/deep-learning-with-pytorch/part-1-core-pytorch/real-world-data-representation-using-tensors.html",
            "Real-World Data Representation Using Tensors: Images, Volumetric Data, Tables, Time Series, and Text",
            theme
          ),
          SubHeading(
            "1.5",
            "/deep-learning-with-pytorch/part-1-core-pytorch/the-mechanics-of-learning.html",
            "The Mechanics of Learning: Parameter Estimation, Loss Functions, Autograd, and Optimizers",
            theme
          ),
          SubHeading(
            "1.6",
            "/deep-learning-with-pytorch/part-1-core-pytorch/using-a-neural-network-to-fit-the-data.html",
            "Using a Neural Network to Fit the Data: Artificial Neurons, Activation Functions, and Modular PyTorch Architectures",
            theme
          ),
          SubHeading(
            "1.7",
            "/deep-learning-with-pytorch/part-1-core-pytorch/telling-birds-from-airplanes.html",
            "Telling Birds from Airplanes: Learning from Images",
            theme
          ),
          SubHeading(
            "1.8",
            "/deep-learning-with-pytorch/part-1-core-pytorch/using-convolutions-to-generalize.html",
            "Using Convolutions to Generalize",
            theme
          ),
        ],
        true
      )
    ),
    SubHeadingCollapsible(
      "dlwpt-2",
      "2.",
      "",
      "Practical Applications",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "2.1",
            "/deep-learning-with-pytorch/part-2-practical-applications/how-transformers-work.html",
            "How Transformers Work",
            theme
          ),
          SubHeading(
            "2.2",
            "/deep-learning-with-pytorch/part-2-practical-applications/diffusion-models-for-images.html",
            "Diffusion Models for Images",
            theme
          ),
          SubHeading(
            "2.3",
            "/deep-learning-with-pytorch/part-2-practical-applications/using-pytorch-to-fight-cancer.html",
            "Using PyTorch to Fight Cancer",
            theme
          ),
        ],
        true
      )
    ),
  ])}
    </div>

    `;

  const tocContentTr = `
    ${SubHeading("", "/tr/", "Hoş Geldiniz", theme, "13px", "bold")}
    ${HeadingCollapsible(
    "Machine Learning Specialization",
    "ml-specialization-tr"
  )}
    ${SubHeading("", "/tr/machine-learning-specialization", "İçerik", theme)}
    ${SubHeadingList([
    SubHeading(
      "1.",
      "",
      "Supervised Machine Learning: Regresyon ve Sınıflandırma",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "1.1",
            "/tr/machine-learning-specialization/supervised-machine-learning-regression-and-classification/supervised-unsupervised-learning.html",
            "Supervised ve Unsupervised Learning",
            theme
          ),
          SubHeading(
            "1.2",
            "/tr/machine-learning-specialization/supervised-machine-learning-regression-and-classification/linear-regression-and-cost-function.html",
            "Linear Regresyon ve Maliyet Fonksiyonu",
            theme
          ),
          SubHeading(
            "1.3",
            "/tr/machine-learning-specialization/supervised-machine-learning-regression-and-classification/gradient-descent.html",
            "Gradient Descent",
            theme
          ),
          SubHeading(
            "1.4",
            "/tr/machine-learning-specialization/supervised-machine-learning-regression-and-classification/multiple-features.html",
            "Çoklu Özellikler",
            theme
          ),
          SubHeading(
            "1.5",
            "/tr/machine-learning-specialization/supervised-machine-learning-regression-and-classification/feature-engineering-and-polynomial-regression.html",
            "Özellik Mühendisliği ve Polinom Regresyonu",
            theme
          ),
          SubHeading(
            "1.6",
            "/tr/machine-learning-specialization/supervised-machine-learning-regression-and-classification/classification-with-logistic-regression.html",
            "Lojistik Regresyon ile Sınıflandırma",
            theme
          ),
          SubHeading(
            "1.7",
            "/tr/machine-learning-specialization/supervised-machine-learning-regression-and-classification/overfitting-and-regularization.html",
            "Overfitting ve Regularizasyon",
            theme
          ),
          SubHeading(
            "1.8",
            "/tr/machine-learning-specialization/supervised-machine-learning-regression-and-classification/scikit-learn-practical-applications.html",
            "Scikit-Learn: Pratik Uygulamalar",
            theme
          ),
        ],
        true
      )
    ),
  ])}
    ${SubHeadingList([
    SubHeading(
      "2.",
      "",
      "İleri Öğrenme Algoritmaları",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "2.1",
            "/tr/machine-learning-specialization/advanced-learning-algorithms/neural-networks-intuition-and-model.html",
            "Sinir Ağları: Sezgi ve Model",
            theme
          ),
          SubHeading(
            "2.2",
            "/tr/machine-learning-specialization/advanced-learning-algorithms/implementation-of-forward-propagation.html",
            "Forward Propagation Uygulaması",
            theme
          ),
          SubHeading(
            "2.3",
            "/tr/machine-learning-specialization/advanced-learning-algorithms/neural-network-training-and-activation-functions.html",
            "Sinir Ağı Eğitimi ve Aktivasyon Fonksiyonları",
            theme
          ),
          SubHeading(
            "2.4",
            "/tr/machine-learning-specialization/advanced-learning-algorithms/optimizers-and-layer-types.html",
            "Optimizasyon ve Katman Türleri",
            theme
          ),
          SubHeading(
            "2.5",
            "/tr/machine-learning-specialization/advanced-learning-algorithms/model-evaluation-selection-and-improvement.html",
            "Model Değerlendirme, Seçim ve İyileştirme",
            theme
          ),
          SubHeading(
            "2.6",
            "/tr/machine-learning-specialization/advanced-learning-algorithms/decision-trees.html",
            "Karar Ağaçları",
            theme
          ),
        ],
        true
      )
    ),
    SubHeading(
      "3.",
      "",
      "Unsupervised Learning, Öneri Sistemleri, Pekiştirmeli Öğrenme",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "3.1",
            "",
            "Unsupervised Learning",
            theme,
            "13px",
            "normal",
            SubHeadingList(
              [
                SubHeading(
                  "3.1.1",
                  "/tr/machine-learning-specialization/unsupervised-learning-recommenders-reinforcement-learning/k-means-clustering.html",
                  "K-Means Kümeleme",
                  theme
                ),
                SubHeading(
                  "3.1.2",
                  "/tr/machine-learning-specialization/unsupervised-learning-recommenders-reinforcement-learning/anomaly-detection.html",
                  "Anomali Tespiti",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "3.2",
            "/tr/machine-learning-specialization/unsupervised-learning-recommenders-reinforcement-learning/recommender-systems.html",
            "Öneri Sistemleri",
            theme
          ),
          SubHeading(
            "3.3",
            "/tr/machine-learning-specialization/unsupervised-learning-recommenders-reinforcement-learning/reinforcement-learning.html",
            "Pekiştirmeli Öğrenme",
            theme
          ),
        ],
        true
      )
    ),
  ])}
    </div>

    ${HeadingCollapsible("Deep Learning Specialization", "dl-specialization-tr")}
    ${SubHeading("", "/tr/deep-learning-specialization", "İçerik", theme)}
      ${SubHeadingList([
    SubHeading(
      "1.",
      "",
      "Konvolüsyonel Sinir Ağları",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "1.1",
            "",
            "Konvolüsyonel Sinir Ağlarının Temelleri",
            theme,
            "11px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "1.1.1",
                  "/tr/deep-learning-specialization/convolutional-neural-networks/computer-vision-and-edge-detection.html",
                  "Bilgisayarlı Görü ve Kenar Tespiti",
                  theme
                ),
                SubHeading(
                  "1.1.2",
                  "/tr/deep-learning-specialization/convolutional-neural-networks/convolutional-operations.html",
                  "Konvolüsyonel İşlemler",
                  theme
                ),
                SubHeading(
                  "1.1.3",
                  "/tr/deep-learning-specialization/convolutional-neural-networks/cnn-architecture-and-examples.html",
                  "CNN Mimarisi ve Örnekler",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "1.2",
            "",
            "Derin Konvolüsyonel Modeller: Vaka Çalışmaları",
            theme,
            "11px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "1.2.1",
                  "/tr/deep-learning-specialization/convolutional-neural-networks/classic-networks-lenet-alexnet-vgg.html",
                  "Klasik Ağlar: LeNet-5, AlexNet, VGG",
                  theme
                ),
                SubHeading(
                  "1.2.2",
                  "/tr/deep-learning-specialization/convolutional-neural-networks/modern-cnn-architectures-resnet-inception-mobilenet-efficenet.html",
                  "Modern CNN Mimarileri: ResNet, Inception, MobileNet, EfficientNet",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "1.3",
            "",
            "Nesne Tespiti ve Yüz Tanıma",
            theme,
            "11px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "1.3.1",
                  "/tr/deep-learning-specialization/convolutional-neural-networks/object-localization-and-detection.html",
                  "Nesne Lokalizasyonu ve Tespiti",
                  theme
                ),
                SubHeading(
                  "1.3.2",
                  "/tr/deep-learning-specialization/convolutional-neural-networks/evaluation-and-optimization-iou-nms-anchor-boxes.html",
                  "Değerlendirme ve Optimizasyon: IoU, Non-max Suppression, Anchor Boxes",
                  theme
                ),
                SubHeading(
                  "1.3.3",
                  "/tr/deep-learning-specialization/convolutional-neural-networks/region-proposals-and-semantic-segmentation-unet.html",
                  "Bölge Önerileri ve Semantik Segmentasyon: U-Net",
                  theme
                ),
                SubHeading(
                  "1.3.4",
                  "/tr/deep-learning-specialization/convolutional-neural-networks/face-recognition-and-neural-style-transfer.html",
                  "Yüz Tanıma ve Neural Style Transfer",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "1.4",
            "",
            "Sequence Modelleri",
            theme,
            "11px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "1.4.1",
                  "/tr/deep-learning-specialization/sequence-models/recurrent-neural-networks.html",
                  "Tekrarlayan Sinir Ağları (RNN)",
                  theme
                ),
                SubHeading(
                  "1.4.2",
                  "/tr/deep-learning-specialization/sequence-models/natural-language-processing-and-word-embeddings.html",
                  "Doğal Dil İşleme ve Kelime Gömmeleri",
                  theme
                ),
              ],
              true
            )
          ),
        ],
        true
      )
    ),
  ])}
    </div>

    ${HeadingCollapsible("First Principles of Computer Vision", "fpcv-specialization-tr")} 
    ${SubHeading("", "/tr/first-principles-of-computer-vision", "İçerik", theme)}
      ${SubHeadingList([
    SubHeadingCollapsible(
      "fpcv-tr-1",
      "1.",
      "",
      "Giriş (Introduction)",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "1.1",
            "/tr/first-principles-of-computer-vision/introduction-to-computer-vision.html",
            "Bilgisayarlı Görmeye Giriş",
            theme
          ),
        ],
        true
      )
    ),
    SubHeadingCollapsible(
      "fpcv-tr-2",
      "2.",
      "",
      "Görüntüleme (Imaging)",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "2.1",
            "",
            "Görüntü Oluşumu (Image Formation)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "2.1.1",
                  "/tr/first-principles-of-computer-vision/imaging/image-formation/pinhole-and-perspective-projection.html",
                  "İğne Deliği Kamera Modeli ve Perspektif İzdüşüm",
                  theme
                ),
                SubHeading(
                  "2.1.2",
                  "/tr/first-principles-of-computer-vision/imaging/image-formation/lenses-and-depth-of-field.html",
                  "Mercek Sistemleri ve Alan Derinliği",
                  theme
                ),
                SubHeading(
                  "2.1.3",
                  "/tr/first-principles-of-computer-vision/imaging/image-formation/advanced-optical-systems.html",
                  "Gelişmiş Optik Sistemler",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "2.2",
            "",
            "Görüntü Algılama (Image Sensing)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "2.2.1",
                  "/tr/first-principles-of-computer-vision/imaging/image-sensing/overview-history-and-sensor-types.html",
                  "Genel Bakış, Tarihçe ve Görüntü Sensör Türleri",
                  theme
                ),
                SubHeading(
                  "2.2.2",
                  "/tr/first-principles-of-computer-vision/imaging/image-sensing/resolution-noise-and-color-sensing.html",
                  "Çözünürlük, Gürültü, Dinamik Aralık ve Renk Algılama",
                  theme
                ),
                SubHeading(
                  "2.2.3",
                  "/tr/first-principles-of-computer-vision/imaging/image-sensing/camera-response-hdr-and-nature-sensors.html",
                  "Kamera Yanıtı, HDR Görüntüleme ve Doğadaki Sensörler",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "2.3",
            "",
            "İkili Görüntüler (Binary Images)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "2.3.1",
                  "/tr/first-principles-of-computer-vision/imaging/binary-images/overview-and-geometric-properties.html",
                  "Genel Bakış ve Geometrik Özellikler",
                  theme
                ),
                SubHeading(
                  "2.3.2",
                  "/tr/first-principles-of-computer-vision/imaging/binary-images/segmenting-binary-images-and-iterative-modification.html",
                  "İkili Görüntü Segmentasyonu ve İteratif Değişiklikler",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "2.4",
            "",
            "Görüntü İşleme I (Image Processing I)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "2.4.1",
                  "/tr/first-principles-of-computer-vision/imaging/image-processing-1/pixel-processing-lsis-and-convolution.html",
                  "Piksel İşleme, LSIS ve Sürekli Konvolüsyon",
                  theme
                ),
                SubHeading(
                  "2.4.2",
                  "/tr/first-principles-of-computer-vision/imaging/image-processing-1/linear-and-non-linear-filters.html",
                  "Doğrusal ve Doğrusal Olmayan Görüntü Filtreleri",
                  theme
                ),
                SubHeading(
                  "2.4.3",
                  "/tr/first-principles-of-computer-vision/imaging/image-processing-1/template-matching.html",
                  "Şablon Eşleme (Template Matching)",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "2.5",
            "",
            "Görüntü İşleme II (Image Processing II)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "2.5.1",
                  "/tr/first-principles-of-computer-vision/imaging/image-processing-2/overview-fourier-transform-and-convolution-theorem.html",
                  "Genel Bakış, Fourier Dönüşümü ve Konvolüsyon Teoremi",
                  theme
                ),
                SubHeading(
                  "2.5.2",
                  "/tr/first-principles-of-computer-vision/imaging/image-processing-2/filtering-in-frequency-domain-and-deconvolution.html",
                  "Frekans Etki Alanında Filtreleme ve Dekonvolüsyon",
                  theme
                ),
                SubHeading(
                  "2.5.3",
                  "/tr/first-principles-of-computer-vision/imaging/image-processing-2/sampling-theory-and-aliasing.html",
                  "Örnekleme Teorisi ve Aliasing",
                  theme
                ),
              ],
              true
            )
          ),
        ],
        true
      )
    ),
    SubHeadingCollapsible(
      "fpcv-tr-3",
      "3.",
      "",
      "Özellikler ve Sınırlar (Features and Boundaries)",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "3.1",
            "",
            "Kenar Tespiti (Edge Detection)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "3.1.1",
                  "/tr/first-principles-of-computer-vision/features/edge-detection/overview-gradients-and-laplacian.html",
                  "Genel Bakış, Gradyanlar ve Laplacian ile Kenar Tespiti",
                  theme
                ),
                SubHeading(
                  "3.1.2",
                  "/tr/first-principles-of-computer-vision/features/edge-detection/canny-and-corner-detection.html",
                  "Canny Kenar Tespiti ve Köşe Tespiti",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "3.2",
            "",
            "Sınır Tespiti (Boundary Detection)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "3.2.1",
                  "/tr/first-principles-of-computer-vision/features/boundary-detection/overview-fitting-lines-and-active-contours.html",
                  "Genel Bakış, Doğru ve Eğri Uydurma, Aktif Konturlar",
                  theme
                ),
                SubHeading(
                  "3.2.2",
                  "/tr/first-principles-of-computer-vision/features/boundary-detection/hough-transform-and-generalized-hough-transform.html",
                  "Hough Dönüşümü ve Genelleştirilmiş Hough Dönüşümü",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "3.3",
            "",
            "SIFT Tespiti (SIFT Detector)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "3.3.1",
                  "/tr/first-principles-of-computer-vision/features/sift-detector/sift-detector.html",
                  "SIFT Tespiti ve Tanımlayıcı (SIFT Detector and Descriptor)",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "3.4",
            "",
            "Görüntü Birleştirme (Image Stitching)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "3.4.1",
                  "/tr/first-principles-of-computer-vision/features/image-stitching/overview-and-image-transformations.html",
                  "Genel Bakış ve Görüntü Dönüşümleri",
                  theme
                ),
                SubHeading(
                  "3.4.2",
                  "/tr/first-principles-of-computer-vision/features/image-stitching/homography-ransac-warping-and-blending.html",
                  "Homografi, RANSAC, Görüntü Eğme ve Harmanlama",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "3.5",
            "",
            "Yüz Tespiti (Face Detection)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "3.5.1",
                  "/tr/first-principles-of-computer-vision/features/face-detection/face-detection.html",
                  "Yüz Tespiti (Face Detection)",
                  theme
                ),
              ],
              true
            )
          ),

        ],
        true
      )
    ),
    SubHeadingCollapsible(
      "fpcv-tr-4",
      "4.",
      "",
      "3D Yeniden Yapılandırma - Tek Bakış Açısı (3D Reconstruction - Single Viewpoint)",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "4.1",
            "",
            "Radyometri ve Yansıma (Radiometry and Reflectance)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "4.1.1",
                  "/tr/first-principles-of-computer-vision/reconstruction-i/radiometry-and-reflectance/overview-radiometric-concepts-and-brdf.html",
                  "Genel Bakış, Radyometrik Kavramlar, Işınım ve BRDF",
                  theme
                ),
                SubHeading(
                  "4.1.2",
                  "/tr/first-principles-of-computer-vision/reconstruction-i/radiometry-and-reflectance/reflectance-models-rough-surfaces-and-dichromatic-model.html",
                  "Yansıma Modelleri, Pürüzlü Yüzeyler ve Dikromatik Model",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "4.2",
            "",
            "Fotometrik Stereo (Photometric Stereo)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "4.2.1",
                  "/tr/first-principles-of-computer-vision/reconstruction-i/photometric-stereo/overview-gradient-space-and-lambertian-case.html",
                  "Genel Bakış, Gradyan Uzayı, Yansıtma Haritası ve Lambertian Durumu",
                  theme
                ),
                SubHeading(
                  "4.2.2",
                  "/tr/first-principles-of-computer-vision/reconstruction-i/photometric-stereo/calibration-shape-from-normals-and-interreflections.html",
                  "Kalibrasyon Tabanlı Fotometrik Stereo, Normalden Şekil Çıkarma ve İç Yansımalar",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "4.3",
            "",
            "Gölgelendirmeden Şekil Çıkarma (Shape from Shading)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "4.3.1",
                  "/tr/first-principles-of-computer-vision/reconstruction-i/shape-from-shading/shape-from-shading.html",
                  "Genel Bakış, İnsan Algısı, Stereografik İzdüşüm, SfS Algoritması ve İllüzyonlar",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "4.4",
            "",
            "Odaktan ve Odak Kusurundan Derinlik Çıkarma (Depth from Defocus)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "4.4.1",
                  "/tr/first-principles-of-computer-vision/reconstruction-i/depth-from-defocus/depth-from-defocus.html",
                  "Odaktan ve Odak Kusurundan Derinlik Çıkarma (Depth from Focus & Defocus)",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "4.5",
            "",
            "Aktif Aydınlatma Yöntemleri (Active Illumination Methods)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "4.5.1",
                  "/tr/first-principles-of-computer-vision/reconstruction-i/active-illumination/overview-photometric-stereo-and-structured-light-range-finding.html",
                  "Genel Bakış, Fotometrik Stereo Sistemleri ve Yapılandırılmış Işık ile Mesafe Ölçümü",
                  theme
                ),
                SubHeading(
                  "4.5.2",
                  "/tr/first-principles-of-computer-vision/reconstruction-i/active-illumination/phase-shifting-structured-light-and-time-of-flight.html",
                  "Faz Kaydırma Yöntemi, Yapılandırılmış Işık Sistemleri ve Uçuş Süresi Yöntemi",
                  theme
                ),
              ],
              true
            )
          ),
        ],
        true
      )
    ),
    SubHeadingCollapsible(
      "fpcv-tr-5",
      "5.",
      "",
      "3D Yeniden Yapılandırma - Çoklu Bakış Açısı (3D Reconstruction - Multiple Viewpoints)",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "5.1",
            "",
            "Kamera Kalibrasyonu (Camera Calibration)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "5.1.1",
                  "/tr/first-principles-of-computer-vision/reconstruction-ii/camera-calibration/camera-models-and-calibration.html",
                  "Kamera Modelleri, Koordinat Sistemleri ve Kamera Kalibrasyonu",
                  theme
                ),
                SubHeading(
                  "5.1.2",
                  "/tr/first-principles-of-computer-vision/reconstruction-ii/camera-calibration/simple-stereo.html",
                  "Basit Stereo (Simple Stereo)",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "5.2",
            "",
            "Kalibre Edilmemiş Stereo (Uncalibrated Stereo)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "5.2.1",
                  "/tr/first-principles-of-computer-vision/reconstruction-ii/camera-calibration/uncalibrated-stereo.html",
                  "Kalibre Edilmemiş Stereo ve Doğada Stereo Görüş (Uncalibrated Stereo & Stereopsis)",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "5.3",
            "",
            "Optik Akış (Optical Flow)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "5.3.1",
                  "/tr/first-principles-of-computer-vision/reconstruction-ii/camera-calibration/optical-flow.html",
                  "Optik Akış ve Görüntü Hareket Analizi (Optical Flow and Motion Analysis)",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "5.4",
            "",
            "Hareketten Yapı Çıkarma (Structure from Motion)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "5.4.1",
                  "/tr/first-principles-of-computer-vision/reconstruction-ii/camera-calibration/structure-from-motion.html",
                  "Hareketten Yapı Çıkarma ve Tomasi-Kanade Faktörizasyonu (Structure from Motion & Factorization)",
                  theme
                ),
              ],
              true
            )
          ),
        ],
        true
      )
    ),
    SubHeadingCollapsible(
      "fpcv-6-tr",
      "6.",
      "",
      "Algılama (Perception)",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "6.1",
            "",
            "Nesne Takibi (Object Tracking)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "6.1.1",
                  "/tr/first-principles-of-computer-vision/perception/object-tracking/object-tracking.html",
                  "Nesne Takibi ve Arka Plan Çıkarma (Object Tracking & Background Subtraction)",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "6.2",
            "",
            "Görüntü Bölütleme (Image Segmentation)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "6.2.1",
                  "/tr/first-principles-of-computer-vision/perception/image-segmentation/image-segmentation.html",
                  "Görüntü Bölütleme Teknolojileri ve Kümeleme Matematiği",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "6.4",
            "",
            "Görünüm Eşleştirme (Appearance Matching)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "6.4.1",
                  "/tr/first-principles-of-computer-vision/perception/appearance-matching/appearance-representation-and-pca.html",
                  "Görünüm Tabanlı Temsil ve PCA Matematiği",
                  theme
                ),
                SubHeading(
                  "6.4.2",
                  "/tr/first-principles-of-computer-vision/perception/appearance-matching/svd-parametric-manifolds-and-appearance-matching.html",
                  "SVD Optimizasyonu, Parametrik Manifoldlar ve Görünüm Eşleştirme",
                  theme
                ),
              ],
              true
            )
          ),
          SubHeading(
            "6.5",
            "",
            "Yapay Sinir Ağları (Neural Networks)",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "6.5.1",
                  "/tr/first-principles-of-computer-vision/perception/neural-networks/perceptron-and-activation-functions.html",
                  "Perceptron ve Aktivasyon Fonksiyonları",
                  theme
                ),
                SubHeading(
                  "6.5.2",
                  "/tr/first-principles-of-computer-vision/perception/neural-networks/multilayer-neural-networks-and-backpropagation.html",
                  "Çok Katmanlı Ağlar, Gradyan Azalma ve Geriye Yayılım",
                  theme
                ),
              ],
              true
            )
          ),
        ],
        true
      )
    ),
  ])}
    </div>

    ${HeadingCollapsible("Deep Learning with PyTorch", "dlwpt-project-tr")} 
    ${SubHeading("", "/tr/deep-learning-with-pytorch", "İçerik", theme)}
    ${SubHeadingList([
    SubHeadingCollapsible(
      "dlwpt-1-tr",
      "1.",
      "",
      "PyTorch'un Temelleri",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "1.1",
            "/tr/deep-learning-with-pytorch/part-1-core-pytorch/introducing-deep-learning-and-the-pytorch-library.html",
            "Derin Öğrenmeye Giriş ve PyTorch Kütüphanesi",
            theme
          ),
          SubHeading(
            "1.2",
            "/tr/deep-learning-with-pytorch/part-1-core-pytorch/pretrained-networks-and-model-zoo.html",
            "Önceden Eğitilmiş Ağlar ve Model Zoo",
            theme
          ),
          SubHeading(
            "1.3",
            "/tr/deep-learning-with-pytorch/part-1-core-pytorch/it-starts-with-a-tensor.html",
            "Tensörlerle Başlamak: Storage, Strides ve Bellek Mimarisi",
            theme
          ),
          SubHeading(
            "1.4",
            "/tr/deep-learning-with-pytorch/part-1-core-pytorch/real-world-data-representation-using-tensors.html",
            "Tensörlerle Gerçek Dünya Verilerini Temsil Etme: Görüntü, Hacimsel Veri, Tablo, Zaman Serisi ve Metin",
            theme
          ),
          SubHeading(
            "1.5",
            "/tr/deep-learning-with-pytorch/part-1-core-pytorch/the-mechanics-of-learning.html",
            "Öğrenmenin Mekaniği: Parametre Tahmini, Kayıp Fonksiyonları, Autograd ve Optimizatörler",
            theme
          ),
          SubHeading(
            "1.6",
            "/tr/deep-learning-with-pytorch/part-1-core-pytorch/using-a-neural-network-to-fit-the-data.html",
            "Veriye Uydurmak İçin Bir Yapay Sinir Ağı Kullanmak: Yapay Nöronlar, Aktivasyon Fonksiyonları ve Modüler PyTorch Mimarisi",
            theme
          ),
          SubHeading(
            "1.7",
            "/tr/deep-learning-with-pytorch/part-1-core-pytorch/telling-birds-from-airplanes.html",
            "Kuşları Uçaklardan Ayırmak: Görüntülerden Öğrenme",
            theme
          ),
          SubHeading(
            "1.8",
            "/tr/deep-learning-with-pytorch/part-1-core-pytorch/using-convolutions-to-generalize.html",
            "Genelleme Yapmak İçin Konvolüsyonları Kullanmak",
            theme
          ),
        ],
        true
      )
    ),
    SubHeadingCollapsible(
      "dlwpt-tr-2",
      "2.",
      "",
      "Pratik Uygulamalar",
      theme,
      "13px",
      "bold",
      SubHeadingList(
        [
          SubHeading(
            "2.1",
            "/tr/deep-learning-with-pytorch/part-2-practical-applications/how-transformers-work.html",
            "Transformer Mimarisi Nasıl Çalışır?",
            theme
          ),
          SubHeading(
            "2.2",
            "/tr/deep-learning-with-pytorch/part-2-practical-applications/diffusion-models-for-images.html",
            "Görüntüler İçin Difüzyon Modelleri",
            theme
          ),
          SubHeading(
            "2.3",
            "/tr/deep-learning-with-pytorch/part-2-practical-applications/using-pytorch-to-fight-cancer.html",
            "Kanserle Savaşmak İçin PyTorch Kullanımı",
            theme
          ),
        ],
        true
      )
    ),
  ])}
    </div>

    `;

  const tocContent = url.includes("/tr") ? tocContentTr : tocContentEn;

  tocElement.innerHTML = tocContent;
}

function currentUiTheme() {
  var t = null;
  try {
    t = localStorage.getItem("mdbook-theme");
  } catch (e) { }
  if (t) return t;
  var names = ["light", "rust", "coal", "navy", "ayu"];
  for (var i = 0; i < names.length; i++) {
    if (document.documentElement.classList.contains(names[i])) {
      return names[i];
    }
  }
  return "rust";
}

function initializeTOC() {
  updateTOC(currentURL, currentUiTheme());
}

initializeTOC();
