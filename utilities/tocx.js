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
      baseColor: "hsl(0, 0%, 0%)",
      activeColor: "#1f1fff",
      hoverColor: "#5c66d6",
    },
    navy: {
      baseColor: "#bcbdd0",
      activeColor: "#2b79a2",
      hoverColor: "#b7b9cc",
    },
    rust: {
      baseColor: "#bcbdd0",
      activeColor: "#e69f67",
      hoverColor: "#e8aa2e",
    },
  };
  return themes[theme] || themes.light;
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
    ${Heading("Number Theory and Cryptography")}
    ${SubHeading("", "/number-theory-cryptography", "Content", theme)}
    ${SubHeadingList([
      SubHeading(
        "1.",
        "",
        "Modular Arithmetic",
        theme,
        "13px",
        "bold",
        SubHeadingList(
          [
            SubHeading(
              "1.1",
              "/number-theory-cryptography/modular-arithmetic/numbers.html",
              "Numbers",
              theme
            ),
            SubHeading(
              "1.2",
              "/number-theory-cryptography/modular-arithmetic/divisibility.html",
              "Divisibility",
              theme
            ),
            SubHeading(
              "1.3",
              "/number-theory-cryptography/modular-arithmetic/remainders.html",
              "Remainders",
              theme
            ),
            SubHeading(
              "1.4",
              "/number-theory-cryptography/modular-arithmetic/modular-arithmetic.html",
              "Modular Arithmetic",
              theme
            ),
            SubHeading(
              "1.5",
              "/number-theory-cryptography/modular-arithmetic/modular-subraction-and-division.html",
              "Modular Subraction and Division",
              theme
            ),
            SubHeading(
              "1.6",
              "/number-theory-cryptography/modular-arithmetic/questions.html",
              "Questions",
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
            "Euclid's Algorithm",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "2.1",
                  "/number-theory-cryptography/euclids-algorithm/greatest-common-divisor.html",
                  "Greatest Common Divisor",
                  theme
                ),
                SubHeading(
                  "2.2",
                  "/number-theory-cryptography/euclids-algorithm/euclids-algorithm.html",
                  "Euclid's Algorithm",
                  theme
                ),
                SubHeading(
                  "2.3",
                  "/number-theory-cryptography/euclids-algorithm/extended-euclids-algorithm.html",
                  "Extended Euclid's Algorithm",
                  theme
                ),
                SubHeading(
                  "2.4",
                  "/number-theory-cryptography/euclids-algorithm/applications/index.html",
                  "Applications",
                  theme,
                  "12px",
                  "normal",
                  SubHeadingList(
                    [
                      SubHeading(
                        "2.4.1",
                        "/number-theory-cryptography/euclids-algorithm/applications/least-common-multiple.html",
                        "Least Common Multiple",
                        theme
                      ),
                      SubHeading(
                        "2.4.2",
                        "/number-theory-cryptography/euclids-algorithm/applications/diophantine-theorem.html",
                        "Diophantine Theorem",
                        theme
                      ),
                      SubHeading(
                        "2.4.3",
                        "/number-theory-cryptography/euclids-algorithm/applications/modular-division.html",
                        "Modular Division",
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
          SubHeading(
            "3.",
            "",
            "Building Blocks of Cryptography",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "3.1",
                  "/number-theory-cryptography/building-blocks-of-cryptography/integer-factorization/index.html",
                  "Integer Factorization",
                  theme,
                  "12px",
                  "normal",
                  SubHeadingList(
                    [
                      SubHeading(
                        "3.1.1",
                        "/number-theory-cryptography/building-blocks-of-cryptography/integer-factorization/prime-numbers.html",
                        "Prime Numbers",
                        theme
                      ),
                      SubHeading(
                        "3.1.2",
                        "/number-theory-cryptography/building-blocks-of-cryptography/integer-factorization/factoring.html",
                        "Factoring",
                        theme
                      ),
                    ],
                    true
                  )
                ),
                SubHeading(
                  "3.2",
                  "/number-theory-cryptography/building-blocks-of-cryptography/chinese-remainder-theorem.html",
                  "Chinese Remainder Theorem",
                  theme
                ),
                SubHeading(
                  "3.3",
                  "",
                  "Modular Exponentiation",
                  theme,
                  "12px",
                  "normal",
                  SubHeadingList(
                    [
                      SubHeading(
                        "3.3.1",
                        "/number-theory-cryptography/building-blocks-of-cryptography/modular-exponentiation/fast-modular-exponentiation.html",
                        "Fast Modular Exponentiation",
                        theme
                      ),
                      SubHeading(
                        "3.3.2",
                        "/number-theory-cryptography/building-blocks-of-cryptography/modular-exponentiation/fermats-little-theorem.html",
                        "Fermat's Little Theorem",
                        theme
                      ),
                      SubHeading(
                        "3.3.3",
                        "/number-theory-cryptography/building-blocks-of-cryptography/modular-exponentiation/eulers-theorem.html",
                        "Euler's Theorem",
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
          SubHeading(
            "4.",
            "",
            "Cryptography",
            theme,
            "13px",
            "bold",
            SubHeadingList(
              [
                SubHeading(
                  "4.1",
                  "/number-theory-cryptography/cryptography/history-of-cryptography.html",
                  "History of Cryptography",
                  theme
                ),
                SubHeading(
                  "4.2",
                  "/number-theory-cryptography/cryptography/one-time-pad.html",
                  "One-Time Pad",
                  theme
                ),
                SubHeading(
                  "4.3",
                  "/number-theory-cryptography/cryptography/many-time-pad-attack.html",
                  "Many Time Pad Attack",
                  theme
                ),
                SubHeading(
                  "4.4",
                  "/number-theory-cryptography/cryptography/rsa-cryptosystem.html",
                  "RSA Cryptosystem",
                  theme,
                  "12px",
                  "normal",
                  SubHeadingList(
                    [
                      SubHeading(
                        "4.4.1",
                        "/number-theory-cryptography/cryptography/rsa-cryptosystem/attacks-and-vulnerabilities.html",
                        "Attacks and Vulnerabilities",
                        theme
                      ),
                      SubHeading(
                        "4.4.2",
                        "/number-theory-cryptography/cryptography/rsa-cryptosystem/randomness-generation.html",
                        "Randomness Generation",
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
        ])}`;

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
  const theme = localStorage.getItem("mdbook-theme") || "light";
  console.log("theme:", theme);
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
