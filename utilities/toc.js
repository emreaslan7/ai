const currentURL = window.location.href;

console.log("Current URL:", currentURL);

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
  const tocElement = document.querySelector("#sidebar .sidebar-scrollbox"); // TOC element

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
                  ],
                  true
                )
              ),
            ],
            true
          )
        ),
      ])}

    `;

  const tocContentTr = `
    ${Heading("1. Lineer Cebir")}
    ${SubHeadingList([
      SubHeading("1.1", "/index.html", "Açiklama", theme),
      SubHeading(
        "1.2",
        "/tr/bolum1.html",
        "Lineer Cebir İcerigi",
        theme,
        SubHeadingList(
          [
            SubHeading("1.2.1", "/tr/subsection1.html", "Alt Bölüm 1", theme),
            SubHeading("1.2.2", "/tr/subsection2.html", "Alt Bölüm 2", theme),
          ],
          true
        )
      ),
      SubHeading("1.3", "#section3", "Bölüm 3", theme),
    ])}
    ${Heading("Bölümler")}
    ${SubHeadingList([
      SubHeading("2.1", "#section1", "Bölüm 1", theme),
      SubHeading("2.2", "#section2", "Bölüm 2", theme),
      SubHeading("2.3", "#section3", "Bölüm 3", theme),
    ])}`;

  const tocContent = url.includes("/tr") ? tocContentTr : tocContentEn;

  tocElement.innerHTML = tocContent;
}

function initializeTOC() {
  const theme = localStorage.getItem("mdbook-theme") || "rust";
  localStorage.setItem("mdbook-theme", theme);
  localStorage.setItem("theme", theme);
  document.documentElement.classList.add("js");
  console.log("themettt:", theme);
  updateTOC(currentURL, theme);
}

initializeTOC();

function loadKaTeXStylesheet() {
  // Create a link element for the KaTeX CSS
  var link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = "https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css";

  // Find the <main> element inside the div with id "content"
  var mainElement = document.querySelector("#content main");

  // Append the link element to the <main> element
  if (mainElement) {
    mainElement.appendChild(link);
  } else {
    console.error("Main element not found inside #content");
  }
}

// Call the function to load the KaTeX stylesheet
loadKaTeXStylesheet();
