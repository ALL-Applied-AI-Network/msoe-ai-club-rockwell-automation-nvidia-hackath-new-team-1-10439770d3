/*
 * Demo 2 — "Generate the Tool": photos / video → 3D model.
 *
 * Real: capture-set loading (images, or frames pulled from a video), the 3D
 * viewer, point sampling, bounding-box measurement, STL import and export.
 * Simulated: the reconstruction itself. Until the photogrammetry backend is
 * connected, every capture set resolves to a stand-in parametric finger.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { STLExporter } from "three/addons/exporters/STLExporter.js";
import { MeshSurfaceSampler } from "three/addons/math/MeshSurfaceSampler.js";

const $ = (id) => document.getElementById(id);
const host = $("viewer");
const thumbs = $("rThumbs");
const runBtn = $("rRun");
const exportBtn = $("rExport");
const stages = [...document.querySelectorAll("#rStages li")];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ───────── Scene ─────────
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(renderer), 0.04).texture;

const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 5000);
camera.position.set(90, 70, 110);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 1.2;

scene.add(new THREE.HemisphereLight(0xdfe9f5, 0x1a2433, 0.6));
const key = new THREE.DirectionalLight(0xffffff, 1.4);
key.position.set(80, 140, 60);
scene.add(key);

const grid = new THREE.GridHelper(200, 20, 0x2a4262, 0x1f3149);
grid.material.transparent = true;
grid.material.opacity = 0.6;
scene.add(grid);

const solidMat = new THREE.MeshStandardMaterial({ color: 0xb9c3ce, metalness: 0.75, roughness: 0.32, transparent: true, opacity: 0 });
const wireMat = new THREE.MeshBasicMaterial({ color: 0x22d3ee, wireframe: true, transparent: true, opacity: 0.55 });
const pointMat = new THREE.PointsMaterial({ size: 0.9, vertexColors: true, transparent: true, opacity: 1, sizeAttenuation: true });

let mesh = null;
let points = null;
let viewMode = "solid";
let reconstructed = false;
let running = false;

// ───────── Stand-in model: custom aluminium finger for the SMC MHZL2-16D2 ─────────
function buildFinger() {
  // Side profile (mm): mounting block on top, long finger below, V-notch at the tip.
  const s = new THREE.Shape();
  s.moveTo(-14, 0);
  s.lineTo(14, 0);
  s.lineTo(14, -9);
  s.lineTo(4, -9);
  s.lineTo(4, -48);
  s.lineTo(1.5, -52);
  s.lineTo(4, -56);
  s.lineTo(4, -60);
  s.lineTo(-4, -60);
  s.lineTo(-4, -9);
  s.lineTo(-14, -9);
  s.closePath();
  for (const x of [-9, 9]) {
    const h = new THREE.Path();
    h.absarc(x, -4.5, 1.7, 0, Math.PI * 2, true); // M3 clearance
    s.holes.push(h);
  }
  const slot = new THREE.Path();
  slot.moveTo(-1.2, -16); slot.lineTo(1.2, -16); slot.lineTo(1.2, -40); slot.lineTo(-1.2, -40); slot.closePath(); // lightening slot
  s.holes.push(slot);

  const geo = new THREE.ExtrudeGeometry(s, { depth: 12, bevelEnabled: true, bevelThickness: 0.6, bevelSize: 0.5, bevelSegments: 2, curveSegments: 24 });
  geo.translate(0, 60, -6);
  geo.computeVertexNormals();
  return geo;
}

function setModel(geometry, { reveal = false } = {}) {
  if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); }
  if (points) { scene.remove(points); points.geometry.dispose(); }
  geometry.computeBoundingBox();
  const bb = geometry.boundingBox;
  const center = bb.getCenter(new THREE.Vector3());
  geometry.translate(-center.x, -bb.min.y, -center.z); // sit on the grid
  geometry.computeBoundingBox();

  mesh = new THREE.Mesh(geometry, solidMat);
  scene.add(mesh);
  points = makePoints(mesh, 26000);
  scene.add(points);
  frame(geometry.boundingBox);

  const size = geometry.boundingBox.getSize(new THREE.Vector3());
  $("dimX").textContent = size.x.toFixed(1);
  $("dimY").textContent = size.y.toFixed(1);
  $("dimZ").textContent = size.z.toFixed(1);
  if (!reveal) { setDrawn(points.geometry.attributes.position.count); applyView(); }
}

function makePoints(m, count) {
  const sampler = new MeshSurfaceSampler(m).build();
  const pos = new Float32Array(count * 3);
  const col = new Float32Array(count * 3);
  const base = new Float32Array(count * 3);
  const p = new THREE.Vector3();
  const bb = m.geometry.boundingBox;
  const a = new THREE.Color(0x76b900), b = new THREE.Color(0x22d3ee), c = new THREE.Color();
  for (let i = 0; i < count; i++) {
    sampler.sample(p);
    base.set([p.x, p.y, p.z], i * 3);
    pos.set([p.x, p.y, p.z], i * 3);
    c.copy(a).lerp(b, (p.y - bb.min.y) / (bb.max.y - bb.min.y || 1));
    col.set([c.r, c.g, c.b], i * 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  g.userData.base = base;
  return new THREE.Points(g, pointMat);
}

function jitterPoints(amount) {
  const pos = points.geometry.attributes.position;
  const base = points.geometry.userData.base;
  for (let i = 0; i < pos.array.length; i++) pos.array[i] = base[i] + (Math.random() - 0.5) * amount;
  pos.needsUpdate = true;
}

function setDrawn(n) {
  points.geometry.setDrawRange(0, n);
  $("ptCount").textContent = n.toLocaleString();
}

function frame(bb) {
  const size = bb.getSize(new THREE.Vector3()).length();
  const center = bb.getCenter(new THREE.Vector3());
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(0.9, 0.55, 1.1).normalize().multiplyScalar(size * 2.3));
  camera.near = size / 100;
  camera.far = size * 20;
  camera.updateProjectionMatrix();
  grid.scale.setScalar(Math.max(0.5, size / 120));
}

// ───────── View modes ─────────
document.querySelectorAll("[data-view]").forEach((btn) => {
  btn.addEventListener("click", () => {
    viewMode = btn.dataset.view;
    document.querySelectorAll("[data-view]").forEach((b) => b.classList.toggle("is-active", b === btn));
    applyView();
  });
});

function applyView() {
  if (!mesh) return;
  mesh.visible = viewMode !== "points";
  mesh.material = viewMode === "wire" ? wireMat : solidMat;
  solidMat.opacity = 1;
  points.visible = viewMode === "points";
  pointMat.opacity = 1;
}

// ───────── Capture set ─────────
function setCapture(kind) {
  $("rSample").classList.toggle("is-active", kind === "sample");
  document.querySelector('label[for="rFile"]').classList.toggle("is-active", kind !== "sample");
}

function renderSampleViews() {
  // Render 12 synthetic views of the stand-in model to act as the sample capture set.
  thumbs.innerHTML = "";
  const g = buildFinger();
  const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: 0xb9c3ce, metalness: 0.75, roughness: 0.32 }));
  g.computeBoundingBox();
  const c = g.boundingBox.getCenter(new THREE.Vector3());
  const s = new THREE.Scene();
  s.environment = scene.environment;
  s.background = new THREE.Color(0xd9dee5);
  s.add(m, new THREE.HemisphereLight(0xffffff, 0x445566, 0.8));
  const cam = new THREE.PerspectiveCamera(35, 1, 1, 1000);
  const prev = new THREE.Vector2();
  renderer.getSize(prev);
  renderer.setSize(160, 160, false);
  for (let i = 0; i < 12; i++) {
    const a = (i / 12) * Math.PI * 2;
    const el = 0.35 + 0.25 * Math.sin(i * 1.3);
    cam.position.set(c.x + Math.cos(a) * 130, c.y + el * 120, c.z + Math.sin(a) * 130);
    cam.lookAt(c);
    renderer.render(s, cam);
    const t = document.createElement("canvas");
    t.width = t.height = 160;
    t.getContext("2d").drawImage(renderer.domElement, 0, 0, 160, 160);
    thumbs.appendChild(t);
  }
  renderer.setSize(prev.x, prev.y, false);
  g.dispose();
  $("rCaption").textContent = "Sample: 12 views of a custom parallel-gripper finger (rendered).";
}

async function loadFiles(files) {
  const list = [...files];
  if (!list.length) return;
  thumbs.innerHTML = "";
  const images = list.filter((f) => f.type.startsWith("image/"));
  const video = list.find((f) => f.type.startsWith("video/"));
  for (const f of images.slice(0, 40)) {
    const img = new Image();
    img.src = URL.createObjectURL(f);
    img.alt = f.name;
    thumbs.appendChild(img);
  }
  if (video) await videoToFrames(video, 12);
  const n = thumbs.children.length;
  $("rCaption").textContent = `Your capture set: ${n} view${n === 1 ? "" : "s"}${video ? " (frames pulled from " + video.name + ")" : ""}. Processed locally.`;
  setCapture("upload");
  resetStages();
}

async function videoToFrames(file, count) {
  const v = document.createElement("video");
  v.muted = true; v.playsInline = true; v.src = URL.createObjectURL(file);
  await new Promise((res, rej) => { v.onloadedmetadata = res; v.onerror = () => rej(new Error("unsupported video")); });
  for (let i = 0; i < count; i++) {
    await new Promise((res) => { v.onseeked = res; v.currentTime = ((i + 0.5) / count) * v.duration; });
    const c = document.createElement("canvas");
    c.width = c.height = 160;
    const side = Math.min(v.videoWidth, v.videoHeight);
    c.getContext("2d").drawImage(v, (v.videoWidth - side) / 2, (v.videoHeight - side) / 2, side, side, 0, 0, 160, 160);
    thumbs.appendChild(c);
  }
}

$("rSample").addEventListener("click", () => { if (running) return; setCapture("sample"); renderSampleViews(); resetStages(); });
$("rFile").addEventListener("change", (e) => { if (!running) loadFiles(e.target.files); e.target.value = ""; });
const drop = $("rDrop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("is-over"); }));
drop.addEventListener("drop", (e) => { if (!running) loadFiles(e.dataTransfer.files); });

// ───────── Reconstruction run (simulated) ─────────
function resetStages() {
  stages.forEach((li) => li.classList.remove("is-running", "is-done"));
  ["rFeat", "rSparse", "rDense"].forEach((id) => ($(id).textContent = "—"));
}
function stage(name, state) {
  const li = stages.find((l) => l.dataset.stage === name);
  li.classList.remove("is-running", "is-done");
  li.classList.add(`is-${state}`);
}

runBtn.addEventListener("click", async () => {
  if (running) return;
  const views = [...thumbs.children];
  if (!views.length) { $("rStatus").textContent = "Add some photos first."; return; }
  running = true;
  runBtn.disabled = true;
  exportBtn.disabled = true;
  runBtn.textContent = "Reconstructing…";
  resetStages();
  const isSample = $("rSample").classList.contains("is-active");
  $("rStatus").textContent = "Running…";

  setModel(buildFinger(), { reveal: true });
  const total = points.geometry.attributes.position.count;
  mesh.visible = false;
  points.visible = true;
  pointMat.opacity = 1;
  setDrawn(0);

  // 1. features — scan each view
  stage("features", "running");
  let feats = 0;
  for (const v of views) {
    v.classList.add("is-scan");
    feats += 1800 + Math.floor(Math.random() * 1400);
    $("rFeat").textContent = `${feats.toLocaleString()} kp`;
    await sleep(Math.max(40, 900 / views.length));
    v.classList.remove("is-scan");
  }
  stage("features", "done");

  // 2. match
  stage("match", "running");
  await sleep(700);
  stage("match", "done");

  // 3. sparse — noisy, few points
  stage("sparse", "running");
  const sparseN = Math.floor(total * 0.12);
  await grow(0, sparseN, 900, 2.4);
  $("rSparse").textContent = `${sparseN.toLocaleString()} pts`;
  stage("sparse", "done");

  // 4. dense — many points, converging
  stage("dense", "running");
  await grow(sparseN, total, 1400, 0.8);
  for (let k = 0; k < 10; k++) { jitterPoints(0.8 * (1 - k / 10)); await sleep(40); }
  jitterPoints(0);
  $("rDense").textContent = `${total.toLocaleString()} pts`;
  stage("dense", "done");

  // 5. mesh — fade surface in, points out
  stage("mesh", "running");
  mesh.visible = true;
  mesh.material = solidMat;
  for (let k = 0; k <= 20; k++) { solidMat.opacity = k / 20; pointMat.opacity = 1 - k / 24; await sleep(35); }
  stage("mesh", "done");
  applyView();

  reconstructed = true;
  exportBtn.disabled = false;
  running = false;
  runBtn.disabled = false;
  runBtn.textContent = "↻ Reconstruct again";
  $("rStatus").textContent = isSample
    ? "Done. Mesh scaled in mm — export it to CAD."
    : "Backend not connected yet: showing the stand-in model for your capture set.";
});

async function grow(from, to, ms, noise) {
  const t0 = performance.now();
  return new Promise((resolve) => {
    const tick = (now) => {
      const f = Math.min(1, (now - t0) / ms);
      setDrawn(Math.floor(from + (to - from) * f));
      jitterPoints(noise);
      if (f < 1) requestAnimationFrame(tick); else resolve();
    };
    requestAnimationFrame(tick);
  });
}

// ───────── STL in / out (real) ─────────
exportBtn.addEventListener("click", () => {
  if (!mesh) return;
  const data = new STLExporter().parse(new THREE.Mesh(mesh.geometry), { binary: true });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([data], { type: "model/stl" }));
  a.download = reconstructed ? "ateam-reconstruction.stl" : "model.stl";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

$("stlFile").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (!f || running) return;
  try {
    const geo = new STLLoader().parse(await f.arrayBuffer());
    geo.computeVertexNormals();
    solidMat.opacity = 1;
    setModel(geo);
    exportBtn.disabled = false;
    $("rStatus").textContent = `Loaded ${f.name} (${geo.attributes.position.count / 3 | 0} triangles). Dimensions assume mm.`;
  } catch (err) {
    $("rStatus").textContent = `Couldn't read that STL: ${err.message}`;
  }
});

// ───────── Render loop ─────────
function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(host);

controls.addEventListener("start", () => (controls.autoRotate = false));
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

renderSampleViews();
resize();
// Start with the point cloud teaser so the viewer isn't empty.
setModel(buildFinger(), { reveal: true });
mesh.visible = false;
points.visible = true;
setDrawn(Math.floor(points.geometry.attributes.position.count * 0.12));
jitterPoints(2.4);
