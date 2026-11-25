const demo = document.querySelector("#demo");
const play = document.querySelector("#play-demo");
demo.controls = false;
play.hidden = false;
play.addEventListener("click", () => {
  play.hidden = true;
  demo.controls = true;
  demo.focus();
  demo.play().catch(() => { /* Native controls remain available for a retry. */ });
});

const motionScene = document.querySelector("#motion-scene");
const motionClips = document.querySelectorAll("[data-motion-scene]");
function stopClip(video) {
  if (!video.paused) video.pause();
  if (video.currentTime > 0) video.currentTime = 0;
}
function selectMotionScene() {
  for (const clip of motionClips) {
    clip.hidden = clip.dataset.motionScene !== motionScene.value;
    if (clip.hidden) stopClip(clip.querySelector("video"));
  }
}
for (const clip of motionClips) {
  const stop = clip.querySelector("button");
  stop.hidden = false;
  stop.addEventListener("click", () => stopClip(clip.querySelector("video")));
}
motionScene.closest("label").hidden = false;
motionScene.addEventListener("change", selectMotionScene);
selectMotionScene();

const gallery = document.querySelector("#gallery-controls");
const scene = gallery.elements.namedItem("scene");
const count = gallery.elements.namedItem("count");
const left = gallery.elements.namedItem("left");
const preset = gallery.elements.namedItem("preset");
const preset2 = gallery.elements.namedItem("preset2");
const layout = gallery.elements.namedItem("layout");
const comparison = document.querySelector(".compare");
const split = document.querySelector("#split");
const split2 = document.querySelector("#split2");
function updateSplit() {
  const triple = count.value === "3";
  const first = split.valueAsNumber;
  const second = triple ? split2.valueAsNumber : 100;
  // Each native range owns the pointer area nearest its divider and stops before crossing the other.
  split.max = triple ? second - 6 : 100;
  split2.min = triple ? first + 6 : 0;
  comparison.style.setProperty("--split", first + "%");
  comparison.style.setProperty("--split2", second + "%");
  comparison.style.setProperty("--first-range", split.max + "%");
  comparison.style.setProperty("--second-start", split2.min + "cqw");
  comparison.style.setProperty("--hit-mid", (first + second) / 2 + "cqw");
  split.setAttribute("aria-valuetext", first + "% from the left");
  split2.setAttribute("aria-valuetext", second + "% from the left");
}
function updateLayout() {
  const triple = count.value === "3";
  const sliding = layout.value === "slider";
  comparison.classList.toggle("is-slider", sliding);
  comparison.classList.toggle("is-triple", triple);
  left.parentElement.hidden = triple;
  preset2.parentElement.hidden = !triple;
  document.querySelector("#render2").parentElement.hidden = !triple;
  document.querySelector("#render2-link").parentElement.hidden = !triple;
  document.querySelector("#preset-label").textContent = triple ? "Preset 1" : "Right";
  split.hidden = !sliding;
  split2.hidden = !sliding || !triple;
  document.querySelector("#split-help").hidden = !sliding;
}
function setPlate(imageId, linkId, source) {
  const name = source === "reference" ? "Reference" : preset.querySelector('option[value="' + source + '"]').text;
  const image = document.getElementById(imageId);
  const link = document.getElementById(linkId);
  const path = "assets/reconstructions/" + scene.value + "-" + source + ".webp";
  if (image.getAttribute("src") !== path) {
    // Browsers may keep the previous bitmap on screen while the new source downloads.
    image.style.visibility = "hidden";
    image.onload = image.onerror = () => { image.style.visibility = ""; };
    image.src = path;
  }
  image.alt = scene.selectedOptions[0].text + " " + name + " image and matching detail crop";
  image.parentElement.href = image.src;
  link.href = image.src;
  link.textContent = name;
  return source === "reference" ? name : name + (source === "quality" ? " (3,500 steps)" : " (3,000 steps)");
}
function showComparison() {
  const triple = count.value === "3";
  for (const option of left.options) option.disabled = option.value === preset.value;
  for (const option of preset.options) option.disabled = option.value === (triple ? preset2.value : left.value);
  for (const option of preset2.options) option.disabled = option.value === preset.value;
  const first = setPlate("reference", "ref-link", triple ? "reference" : left.value);
  const descriptions = [first, setPlate("render", "render-link", preset.value)];
  if (triple) descriptions.push(setPlate("render2", "render2-link", preset2.value));
  document.querySelector("#render-caption").textContent = scene.selectedOptions[0].text + " · " + descriptions.join(" · ");
  split.setAttribute("aria-label", "Divider after " + document.querySelector("#ref-link").textContent);
  split2.setAttribute("aria-label", "Divider after " + preset.selectedOptions[0].text);
}
function changeCount() {
  // The rightmost image stays in place when a third image is inserted or removed.
  const previous = preset.value;
  preset.value = preset2.value;
  preset2.value = previous;
  if (count.value === "3") {
    if (left.value !== "reference") preset.value = left.value;
    if (preset.value === preset2.value) preset.value = preset2.value === "ultrafast" ? "quality" : "ultrafast";
  } else if (left.value === preset.value) {
    left.value = "reference";
  }
  split.max = 100;
  split2.min = 0;
  split.value = count.value === "3" ? 33 : 50;
  split2.value = 67;
  updateSplit();
  updateLayout();
  showComparison();
}
gallery.hidden = false;
updateSplit();
updateLayout();
showComparison();
split.addEventListener("input", updateSplit);
split2.addEventListener("input", updateSplit);
count.addEventListener("change", changeCount);
layout.addEventListener("change", updateLayout);
scene.addEventListener("change", showComparison);
left.addEventListener("change", showComparison);
preset.addEventListener("change", showComparison);
preset2.addEventListener("change", showComparison);
for (const link of document.querySelectorAll("[data-scene]")) {
  link.href = "#reconstructions";
  link.addEventListener("click", () => {
    scene.value = link.dataset.scene;
    if (count.value === "2") {
      left.value = "reference";
      preset.value = "quality";
    }
    showComparison();
  });
}
