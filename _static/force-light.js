document.body.dataset.theme = "light";
try {
  localStorage.setItem("theme", "light");
} catch {
  // Ignore browsers that block localStorage.
}
