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
  const title = element.previousElementSibling;

  if (element.style.display === "none") {
    element.style.display = "block";
    title.innerHTML = `▼ ${title.textContent.trim().replace(/^▶|^▼/, "")}`;
    localStorage.setItem(id, "open"); // Durumu kaydet
  } else {
    element.style.display = "none";
    title.innerHTML = `▶ ${title.textContent.trim().replace(/^▶|^▼/, "")}`;
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

  return `<a href="${href}" style="${baseStyle} ${
    isActive ? activeStyle : ""
  }" onmouseover="this.style.color='${hoverColor}'" onmouseout="this.style.color='${
    isActive ? activeColor : baseColor
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
        SubHeading(
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
          ),
        ),
        SubHeading(
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
        SubHeading(
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
        SubHeading(
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
  } catch (e) {}
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
