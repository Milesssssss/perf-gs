const demo = document.querySelector("#demo");
const play = document.querySelector("#play-demo");
const teasers = [...document.querySelectorAll(".media-grid video")];
const motionScene = document.querySelector("#motion-scene");
const motionClips = document.querySelectorAll("[data-motion-scene]");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const group = new Set();
let cycle = 0;

function startTogether(videos) {
  const members = [...videos].filter(video => !video.error);
  const started = ++cycle;
  group.clear();
  // Rewind every participant before asking any of them to play.
  for (const video of members) {
    group.add(video);
    if (video.currentTime !== 0) video.currentTime = 0;
  }
  for (const video of members) {
    video.play().catch(error => {
      if (started !== cycle || !video.paused) return;
      if (video === demo && error.name === "NotAllowedError") play.hidden = false;
      leaveGroup(video);
    });
  }
}

function restartIfFinished() {
  // An ended video keeps displaying its last frame while its peers finish.
  if (group.size && [...group].every(video => video.ended)) startTogether(group);
}

function leaveGroup(video) {
  if (group.delete(video)) restartIfFinished();
}

for (const video of document.querySelectorAll("video")) {
  video.addEventListener("ended", restartIfFinished);
  video.addEventListener("pause", () => {
    if (video.paused && !video.ended) leaveGroup(video);
  });
  video.addEventListener("play", () => {
    if (!video.paused && !video.closest("[hidden]")) group.add(video);
  });
  video.addEventListener("error", () => leaveGroup(video), true);
}

function stopClip(video) {
  video.pause();
  if (video.currentTime !== 0) video.currentTime = 0;
}

function showSelectedComparison() {
  let selected;
  for (const clip of motionClips) {
    const video = clip.querySelector("video");
    clip.hidden = clip.dataset.motionScene !== motionScene.value;
    if (clip.hidden) {
      group.delete(video);
      stopClip(video);
    } else {
      selected = video;
    }
  }
  return selected;
}

let comparisonVideo = showSelectedComparison();
motionScene.closest("label").hidden = false;
motionScene.addEventListener("change", () => {
  comparisonVideo = showSelectedComparison();
  if (!reducedMotion.matches) {
    startTogether([...teasers.filter(video => group.has(video)), comparisonVideo]);
  }
});
for (const clip of motionClips) {
  const stop = clip.querySelector("button");
  stop.hidden = false;
  stop.addEventListener("click", () => {
    const video = clip.querySelector("video");
    leaveGroup(video);
    stopClip(video);
  });
}

function applyMotionPreference() {
  if (reducedMotion.matches) {
    ++cycle;
    group.clear();
    for (const video of document.querySelectorAll("video")) video.pause();
  } else {
    startTogether([...teasers, comparisonVideo]);
  }
}
reducedMotion.addEventListener("change", applyMotionPreference);
applyMotionPreference();

const replay = document.querySelector("#replay-training");
replay.hidden = false;
replay.addEventListener("click", () => {
  demo.focus({ preventScroll: true });
  demo.scrollIntoView({ block: "start" });
  startTogether([...teasers, comparisonVideo]);
});

demo.addEventListener("playing", () => { play.hidden = true; });
play.addEventListener("click", () => {
  play.hidden = true;
  demo.focus();
  startTogether([...teasers, comparisonVideo]);
});

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
