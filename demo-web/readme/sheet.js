// Shared by the sheets: theme, panel loading and compositing, the ready signal shoot.mjs waits for.
const q = new URLSearchParams(location.search);
export const theme = q.get("theme") === "light" ? "light" : "dark";
if (theme === "light") document.documentElement.classList.add("light");
const css = getComputedStyle(document.documentElement);
export const v = (n) => css.getPropertyValue(n).trim();
export const stats = await fetch("out/readme.json").then((r) => r.json());

export const img = (src) =>
  new Promise((ok, err) => {
    const i = new Image();
    i.onload = () => ok(i);
    i.onerror = () => err(new Error(`missing ${src}`));
    i.src = src;
  });

/** A canvas `w` by `h` CSS pixels at device resolution, its 2D context filled with the panel color. */
export function canvas(w, h, cls = "p") {
  const c = document.createElement("canvas");
  c.width = Math.round(w * devicePixelRatio);
  c.height = Math.round(h * devicePixelRatio);
  c.style.width = `${w}px`;
  c.style.height = `${h}px`;
  c.className = cls;
  const g = c.getContext("2d");
  g.fillStyle = v("--panel");
  g.fillRect(0, 0, c.width, c.height);
  g.imageSmoothingQuality = "high";
  return [c, g];
}

export const int = (n) => Math.round(n).toLocaleString("en-US");
export const pct = (x) => `${(100 * x).toPrecision(2)}%`;

export async function ready() {
  await document.fonts.ready;
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  window.__ready = true;
}
