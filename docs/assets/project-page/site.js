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
const preset = gallery.elements.namedItem("preset");
const preset2 = gallery.elements.namedItem("preset2");
const comparison = document.querySelector(".compare");
const split = document.querySelector("#split");
const split2 = document.querySelector("#split2");
function updateSplit(event) {
  if (split.valueAsNumber > split2.valueAsNumber) {
    if (event?.target === split2) split.value = split2.value;
    else split2.value = split.value;
  }
  const first = split.valueAsNumber;
  const second = split2.valueAsNumber;
  comparison.style.setProperty("--split", first + "%");
  comparison.style.setProperty("--split2", second + "%");
  comparison.style.setProperty("--hit-mid", (first + second) / 2 + "%");
  split.setAttribute("aria-valuetext", first + "% from the left");
  split2.setAttribute("aria-valuetext", second + "% from the left");
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
}
function showComparison() {
  for (const option of preset.options) option.disabled = option.value === preset2.value;
  for (const option of preset2.options) option.disabled = option.value === preset.value;
  setPlate("reference", "ref-link", "reference");
  setPlate("render", "render-link", preset.value);
  setPlate("render2", "render2-link", preset2.value);
  const view = scene.selectedOptions[0].dataset;
  const scores = [preset, preset2].map(select => select.selectedOptions[0].text + " " + view[select.value] + " dB");
  document.querySelector("#render-caption").textContent = [view.photo, ...scores].join(" · ");
  split2.setAttribute("aria-label", "Divider after " + preset.selectedOptions[0].text);
}
comparison.classList.add("is-slider");
gallery.hidden = false;
document.querySelector(".plate-controls").hidden = false;
document.querySelector("#split-help").hidden = false;
split.hidden = false;
split2.hidden = false;
updateSplit();
showComparison();
split.addEventListener("input", updateSplit);
split2.addEventListener("input", updateSplit);
scene.addEventListener("change", showComparison);
preset.addEventListener("change", showComparison);
preset2.addEventListener("change", showComparison);
for (const link of document.querySelectorAll("[data-scene]")) {
  link.href = "#reconstructions";
  link.addEventListener("click", () => {
    scene.value = link.dataset.scene;
    showComparison();
  });
}
