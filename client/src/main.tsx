import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

const bootSplash = document.getElementById("boot-splash");

createRoot(document.getElementById("root")!).render(<App />);

const hideBootSplash = () => {
  if (!bootSplash) {
    return;
  }

  bootSplash.classList.add("boot-splash--hidden");
  window.setTimeout(() => {
    bootSplash.remove();
  }, 260);
};

window.requestAnimationFrame(() => {
  window.requestAnimationFrame(hideBootSplash);
});
