"use strict";

const links = window.INFLUPOSE_LINKS || {};
document.querySelectorAll("[data-resource]").forEach((element) => {
  const value = links[element.dataset.resource];
  if (typeof value !== "string" || !value.trim()) {
    element.hidden = true;
    return;
  }
  const url = value.trim();
  element.hidden = false;
  if (!url.startsWith("https://") && !url.startsWith("assets/")) return;
  element.href = url;
  element.removeAttribute("aria-disabled");
  element.removeAttribute("tabindex");
  element.querySelectorAll("small").forEach((label) => label.remove());
  if (url.startsWith("https://")) {
    element.target = "_blank";
    element.rel = "noopener noreferrer";
  }
});
