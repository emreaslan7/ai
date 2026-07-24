// KaTeX auto-render initialization
// Called after DOM content is loaded to render math expressions
document.addEventListener("DOMContentLoaded", function () {
    if (typeof renderMathInElement === "function") {
        renderMathInElement(document.body, {
            delimiters: [
                { left: "$$", right: "$$", display: true },
                { left: "$", right: "$", display: false },
                { left: "\\(", right: "\\)", display: false },
                { left: "\\[", right: "\\]", display: true }
            ],
            throwOnError: false,
            errorCallback: function (err, node) {
                // Silently ignore errors so broken math doesn't break the page
                console.warn("KaTeX auto-render error:", err);
            }
        });
    } else {
        console.warn("KaTeX auto-render not available — math expressions will display as plain text.");
    }
});
