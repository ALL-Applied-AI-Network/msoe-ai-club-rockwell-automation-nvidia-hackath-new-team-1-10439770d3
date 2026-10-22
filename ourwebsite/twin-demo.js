/*
 * Demo 3 — "Drag to Move": a lightweight replica of the FANUC ER-4iA cell.
 *
 * Parts use the official SIDC CAD models (cubes, M6 screw, bin). The arm is
 * simplified geometry driven by analytic IK with the ER-4iA link lengths.
 * Dragging a part leaves a ghost at the target; the arm then plans a
 * pick-and-place at the cell's 20 mm/s cap and carries it there.
 * Cell frame: origin at the robot base on the table top, +X forward, +Y left,
 * +Z up, millimetres. three.js is Y-up, so (x, y, z)cell → (x, z, −y)three.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const $ = (id) => document.getElementById(id);
const host = $("twin");

// ER-4iA kinematics (scene_config.py) and cell limits (robot-mvp/policy.py).
const SHOULDER_H = 330, L1 = 260, L2 = 290, TOOL = 140;
const SPEED_MM_S = 20;
const MAX_STEP_MM = 2;
const GRIP_DWELL_S = 0.5;
const SAFE_Z = 140;
const HOME = [300, 0, 250];
const TABLE_X = [-371, 759], TABLE_Y = [-772, 792];
const BIN_HALF = 25, BIN_FLOOR = 2;

const toThree = (x, y, z) => new THREE.Vector3(x, z, -y);

// ───────── Renderer / scene ─────────
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(renderer), 0.04).texture;

const camera = new THREE.PerspectiveCamera(38, 1, 5, 8000);
camera.position.copy(toThree(1120, -520, 820));
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(toThree(250, -10, 170));
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.47;
controls.minDistance = 250;
controls.maxDistance = 2600;

scene.add(new THREE.HemisphereLight(0xeef4ff, 0x2a3442, 0.55));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.copy(toThree(500, -300, 1400));
sun.target.position.copy(toThree(350, 0, 0));
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -700, right: 700, top: 700, bottom: -700, near: 200, far: 3000 });
sun.shadow.bias = -0.0004;
scene.add(sun, sun.target);

// Table: white top, yellow caution tape on the front and side edges.
{
  const w = TABLE_X[1] - TABLE_X[0], d = TABLE_Y[1] - TABLE_Y[0];
  const top = new THREE.Mesh(new THREE.BoxGeometry(w, 50, d), new THREE.MeshStandardMaterial({ color: 0xeceff3, roughness: 0.85 }));
  top.position.copy(toThree((TABLE_X[0] + TABLE_X[1]) / 2, (TABLE_Y[0] + TABLE_Y[1]) / 2, -25));
  top.receiveShadow = true;
  scene.add(top);
  const tape = new THREE.MeshStandardMaterial({ color: 0xf2c200, roughness: 0.6 });
  const band = (cx, cy, sx, sy) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(sx, 0.6, sy), tape);
    m.position.copy(toThree(cx, cy, 0.3));
    m.receiveShadow = true;
    scene.add(m);
  };
  band(TABLE_X[1] - 24, (TABLE_Y[0] + TABLE_Y[1]) / 2, 48, d);
  band((TABLE_X[0] + TABLE_X[1]) / 2, TABLE_Y[0] + 24, w, 48);
  band((TABLE_X[0] + TABLE_X[1]) / 2, TABLE_Y[1] - 24, w, 48);
  // Reach ring at the IK limit for floor-height picks.
  const ring = new THREE.Mesh(new THREE.RingGeometry(503, 507, 128, 1, -Math.PI * 0.42, Math.PI * 0.84), new THREE.MeshBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.45 }));
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.8;
  scene.add(ring);
}

// ───────── Robot (simplified ER-4iA) ─────────
const yellow = new THREE.MeshStandardMaterial({ color: 0xf0c018, metalness: 0.15, roughness: 0.45 });
const grey = new THREE.MeshStandardMaterial({ color: 0x80888f, metalness: 0.4, roughness: 0.5 });
const dark = new THREE.MeshStandardMaterial({ color: 0x2b3038, metalness: 0.3, roughness: 0.55 });
const alu = new THREE.MeshStandardMaterial({ color: 0xc2c9d1, metalness: 0.8, roughness: 0.3 });

function part(geo, mat, x = 0, y = 0, z = 0) {
  const m = new THREE.Mesh(geo, mat);
  m.position.set(x, y, z);
  m.castShadow = true;
  m.receiveShadow = true;
  return m;
}
const zCyl = (r, h, seg = 32) => new THREE.CylinderGeometry(r, r, h, seg).rotateX(Math.PI / 2);
const xCyl = (r, h, seg = 32) => new THREE.CylinderGeometry(r, r, h, seg).rotateZ(Math.PI / 2);

const robot = new THREE.Group();
scene.add(robot);
robot.add(part(new THREE.CylinderGeometry(100, 110, 70, 48), grey, 0, 35, 0));
const j1 = new THREE.Group();
j1.position.y = 70;
robot.add(j1);
j1.add(part(new THREE.CylinderGeometry(82, 88, 120, 48), yellow, 0, 60, 0));
j1.add(part(new THREE.BoxGeometry(120, 150, 140), yellow, 0, 185, 0));
const j2 = new THREE.Group();
j2.position.y = SHOULDER_H - 70;
j1.add(j2);
j2.add(part(zCyl(62, 150), dark));
j2.add(part(new THREE.BoxGeometry(L1 + 40, 70, 84).translate(L1 / 2, 0, 0), yellow));
const j3 = new THREE.Group();
j3.position.x = L1;
j2.add(j3);
j3.add(part(zCyl(50, 112), dark));
j3.add(part(new THREE.BoxGeometry(L2, 58, 62).translate(L2 / 2, 0, 0), yellow));
const wrist = new THREE.Group();
wrist.position.x = L2;
j3.add(wrist);
wrist.add(part(zCyl(34, 72), dark));
wrist.add(part(xCyl(26, 12), dark, 40, 0, 0));
wrist.add(part(new THREE.BoxGeometry(46, 30, 42), alu, 68, 0, 0)); // SMC MHZL2-16D2 body
const fingerGeo = new THREE.BoxGeometry(56, 14, 7);
const fingerA = part(fingerGeo, alu, 118, 0, 12);
const fingerB = part(fingerGeo, alu, 118, 0, -12);
wrist.add(fingerA, fingerB);

function solveIK([x, y, z]) {
  const j1a = Math.atan2(y, x);
  const r = Math.hypot(x, y);
  const dx = r, dz = z + TOOL - SHOULDER_H;
  const D = THREE.MathUtils.clamp(Math.hypot(dx, dz), Math.abs(L1 - L2) + 1, L1 + L2 - 1);
  const q = Math.acos(THREE.MathUtils.clamp((D * D - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1, 1));
  const a = Math.atan2(dz, dx) + Math.atan2(L2 * Math.sin(q), L1 + L2 * Math.cos(q));
  const b = a - q;
  return { j1: j1a, a, b };
}
function reachable([x, y, z]) {
  const r = Math.hypot(x, y);
  return r > 190 && Math.hypot(r, z + TOOL - SHOULDER_H) < L1 + L2 - 18;
}

const deg = (r) => (r * 180) / Math.PI;
const jointEls = [...document.querySelectorAll("#tJoints b")];
function poseRobot(tcp, gap) {
  const { j1: t1, a, b } = solveIK(tcp);
  j1.rotation.y = t1;
  j2.rotation.z = a;
  j3.rotation.z = b - a;
  wrist.rotation.z = -Math.PI / 2 - b;
  fingerA.position.z = gap / 2 + 3.5;
  fingerB.position.z = -gap / 2 - 3.5;
  // FANUC-style readout: J2 from vertical, J3 forearm from horizontal, J5 keeps the tool vertical.
  const vals = [deg(t1), 90 - deg(a), deg(b), -90 - deg(b)];
  jointEls.forEach((el, i) => (el.textContent = `${vals[i].toFixed(1)}°`));
  $("tTcp").textContent = tcp.map((v) => v.toFixed(0)).join(", ");
}

// ───────── Parts (official SIDC CAD models) ─────────
const KINDS = {
  red: { file: "manipbench-cube-10mm.stl", color: 0xd8423b, label: "Red cube 10 mm", h: 10, r: 8, grip: 10 },
  yel: { file: "manipbench-cube-16mm.stl", color: 0xf2c94c, label: "Yellow cube 16 mm", h: 16, r: 12, grip: 16 },
  screw: { file: "ISO10642-M6x30-countersunk-screw.stl", color: 0xa7b0ba, label: "M6×30 screw", h: 12, r: 16, grip: 7, metal: true },
};
const LAYOUT = [
  ["red", 360, -95], ["red", 398, -62], ["red", 385, -132], ["red", 425, -100],
  ["yel", 320, 45], ["yel", 300, 100],
  ["screw", 440, 30], ["screw", 470, 62], ["screw", 432, 95], ["screw", 468, 128],
];
const BINS = [
  { id: "binY", label: "yellow bin", x: 420, y: -235, color: 0xe8b923 },
  { id: "binG", label: "green bin", x: 420, y: 235, color: 0x3fa34d },
];

let parts = [];
let tasks = [];
let ghosts = new Map();
let taskSeq = 0;
let simSpeed = 4;
let ready = false;

const loader = new STLLoader();
const geos = {};

async function loadCad() {
  const files = [...new Set([...Object.values(KINDS).map((k) => k.file), "manipbench-bin-50x50x55.stl"])];
  const loaded = await Promise.all(files.map((f) => loader.loadAsync(`assets/cad/${f}`)));
  files.forEach((f, i) => {
    const g = loaded[i];
    if (f.includes("screw")) {
      // Lay the screw on its side, axis along cell +Y, resting on the table.
      g.rotateX(Math.PI / 2);
      g.computeBoundingBox();
      const bb = g.boundingBox;
      g.translate(-(bb.min.x + bb.max.x) / 2, -(bb.min.y + bb.max.y) / 2, -bb.min.z);
    }
    g.rotateX(-Math.PI / 2); // CAD is Z-up
    g.computeVertexNormals();
    geos[f] = g;
  });
}

function buildScene() {
  parts.forEach((p) => scene.remove(p.mesh));
  BINS.forEach((b) => {
    if (!b.mesh) {
      b.mesh = part(geos["manipbench-bin-50x50x55.stl"], new THREE.MeshStandardMaterial({ color: b.color, roughness: 0.55, side: THREE.DoubleSide }));
      b.mesh.position.copy(toThree(b.x, b.y, 0));
      scene.add(b.mesh);
    }
  });
  parts = LAYOUT.map(([kind, x, y], i) => {
    const k = KINDS[kind];
    const mat = new THREE.MeshStandardMaterial({ color: k.color, metalness: k.metal ? 0.85 : 0.05, roughness: k.metal ? 0.3 : 0.5 });
    const mesh = part(geos[k.file], mat);
    const p = { id: `${kind}${i}`, kind, k, x, y, z: 0, bin: null, busy: false, mesh };
    // Real-scale parts are tiny on screen, so grab them through a larger invisible proxy.
    const grab = new THREE.Mesh(new THREE.SphereGeometry(k.r + 12, 12, 8), new THREE.MeshBasicMaterial({ visible: false }));
    grab.position.y = k.h / 2;
    grab.userData.part = p;
    mesh.add(grab);
    p.grab = grab;
    placeMesh(p);
    scene.add(mesh);
    return p;
  });
}
function placeMesh(p, z = p.z) { p.mesh.position.copy(toThree(p.x, p.y, z)); }

// ───────── Ghost targets ─────────
function makeGhost(p) {
  const g = new THREE.Group();
  const fill = new THREE.Mesh(p.mesh.geometry, new THREE.MeshBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.28, depthWrite: false, depthTest: false }));
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(p.mesh.geometry, 30), new THREE.LineBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.95, depthTest: false }));
  const marker = new THREE.Mesh(new THREE.RingGeometry(p.k.r + 4, p.k.r + 7, 40), new THREE.MeshBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.8, side: THREE.DoubleSide }));
  marker.rotation.x = -Math.PI / 2;
  marker.position.y = 0.9;
  fill.renderOrder = edges.renderOrder = 10;
  g.add(fill, edges, marker);
  g.userData = { fill, edges, marker };
  scene.add(g);
  return g;
}
function tintGhost(g, ok) {
  const c = ok ? 0x22d3ee : 0xf36b6b;
  Object.values(g.userData).forEach((m) => m.material.color.setHex(c));
}

function targetsTaken(except) {
  const spots = parts.filter((p) => p !== except && !(p.busy && p.carried)).map((p) => ({ x: p.x, y: p.y, r: p.k.r, bin: p.bin }));
  tasks.filter((t) => t.part !== except && (t.state === "queued" || t.state === "pending" || t.state === "running"))
    .forEach((t) => spots.push({ x: t.to.x, y: t.to.y, r: t.part.k.r, bin: t.to.bin }));
  return spots;
}

function resolveDrop(p, x, y) {
  // Snap into a bin if the drop lands on or near it.
  for (const b of BINS) {
    if (Math.abs(x - b.x) < BIN_HALF + 10 && Math.abs(y - b.y) < BIN_HALF + 10) {
      const slots = p.kind === "screw" ? [[-11, 0], [11, 0], [0, 0]] : [[-11, -11], [11, -11], [-11, 11], [11, 11], [0, 0]];
      const taken = targetsTaken(p).filter((s) => s.bin === b.id);
      const slot = slots.find(([sx, sy]) => taken.every((s) => Math.hypot(s.x - (b.x + sx), s.y - (b.y + sy)) > 10));
      if (!slot) return { x, y, z: BIN_FLOOR, bin: b.id, ok: false, why: `${b.label} is full` };
      return { x: b.x + slot[0], y: b.y + slot[1], z: BIN_FLOOR, bin: b.id, ok: true, where: b.label };
    }
  }
  const to = { x, y, z: 0, bin: null, ok: true, where: "table" };
  const near = (bx, by) => Math.abs(x - bx) < BIN_HALF + p.k.r + 4 && Math.abs(y - by) < BIN_HALF + p.k.r + 4;
  if (x < TABLE_X[0] + 60 || x > TABLE_X[1] - 60 || y < TABLE_Y[0] + 60 || y > TABLE_Y[1] - 60) return { ...to, ok: false, why: "off the table" };
  if (BINS.some((b) => near(b.x, b.y))) return { ...to, ok: false, why: "collides with a bin" };
  if (targetsTaken(p).some((s) => !s.bin && Math.hypot(s.x - x, s.y - y) < s.r + p.k.r)) return { ...to, ok: false, why: "spot is occupied" };
  const gz = to.z + p.k.h / 2;
  if (!reachable([x, y, gz]) || !reachable([x, y, SAFE_Z])) return { ...to, ok: false, why: "outside the arm's reach" };
  return to;
}

// ───────── Dragging ─────────
const ray = new THREE.Raycaster();
const ndc = new THREE.Vector2();
const tablePlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
let drag = null;

function setPointer(e) {
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
}
function hitPart() {
  const hits = ray.intersectObjects(parts.filter((p) => !p.busy).map((p) => p.grab), false);
  return hits[0]?.object.userData.part || null;
}
function tablePoint() {
  const v = new THREE.Vector3();
  return ray.ray.intersectPlane(tablePlane, v) ? { x: v.x, y: -v.z } : null;
}

renderer.domElement.addEventListener("pointerdown", (e) => {
  if (!ready) return;
  setPointer(e);
  const p = hitPart();
  if (!p) return;
  const pt = tablePoint();
  controls.enabled = false;
  renderer.domElement.setPointerCapture(e.pointerId);
  drag = { p, off: pt ? { x: p.x - pt.x, y: p.y - pt.y } : { x: 0, y: 0 }, ghost: makeGhost(p), to: null };
  host.classList.add("is-dragging");
  p.mesh.material.emissive = new THREE.Color(0x22d3ee);
  p.mesh.material.emissiveIntensity = 0.25;
  updateDrag();
});
renderer.domElement.addEventListener("pointermove", (e) => {
  setPointer(e);
  if (drag) return updateDrag();
  if (ready) host.classList.toggle("is-hover", !!hitPart());
});
renderer.domElement.addEventListener("pointerup", (e) => {
  if (!drag) return;
  renderer.domElement.releasePointerCapture(e.pointerId);
  controls.enabled = true;
  host.classList.remove("is-dragging");
  const { p, ghost, to } = drag;
  p.mesh.material.emissiveIntensity = 0;
  drag = null;
  if (!to || Math.hypot(to.x - p.x, to.y - p.y) < 8) { scene.remove(ghost); return; }
  if (!to.ok) {
    scene.remove(ghost);
    status(`Can't place ${p.k.label.toLowerCase()} there: ${to.why}.`);
    return;
  }
  queueTask(p, to, ghost);
});

function updateDrag() {
  const pt = tablePoint();
  if (!pt) return;
  const to = resolveDrop(drag.p, pt.x + drag.off.x, pt.y + drag.off.y);
  drag.to = to;
  drag.ghost.position.copy(toThree(to.x, to.y, to.z));
  tintGhost(drag.ghost, to.ok);
  status(to.ok ? `Target: ${to.where} (${to.x.toFixed(0)}, ${to.y.toFixed(0)})` : `✗ ${to.why}`);
}

// ───────── Tasks ─────────
function queueTask(p, to, ghost) {
  const auto = $("tAuto").checked;
  const t = { id: ++taskSeq, part: p, from: { x: p.x, y: p.y, z: p.z }, to, ghost, state: auto ? "queued" : "pending", realS: 0 };
  t.waypoints = planMove(t);
  p.busy = true;
  tasks.push(t);
  ghosts.set(t.id, ghost);
  status(auto ? `Queued: ${p.k.label} → ${to.where}.` : `Drafted: ${p.k.label} → ${to.where}. Approve it in the queue.`);
  renderQueue();
}

function planMove(t) {
  const k = t.part.k;
  const pz = t.from.z + k.h / 2, dz = t.to.z + k.h / 2;
  const { x: fx, y: fy } = t.from, { x: tx, y: ty } = t.to;
  return [
    { name: "approach", tcp: [fx, fy, SAFE_Z], grip: 0 },
    { name: "descend", tcp: [fx, fy, pz], grip: 0 },
    { name: "grasp", tcp: [fx, fy, pz], grip: 1, dwell: true },
    { name: "lift", tcp: [fx, fy, SAFE_Z], grip: 1 },
    { name: "transfer", tcp: [tx, ty, SAFE_Z], grip: 1 },
    { name: "lower", tcp: [tx, ty, dz], grip: 1 },
    { name: "release", tcp: [tx, ty, dz], grip: 0, dwell: true },
    { name: "retreat", tcp: [tx, ty, SAFE_Z], grip: 0 },
  ];
}

const queueEl = $("tQueue");
function renderQueue() {
  const open = tasks.filter((t) => t.state !== "done").length;
  $("tCount").textContent = `${open} queued`;
  $("tExport").disabled = !tasks.length;
  if (!tasks.length) { queueEl.innerHTML = '<li class="empty">No moves yet. Drag a part in the replica.</li>'; return; }
  queueEl.innerHTML = "";
  [...tasks].reverse().forEach((t) => {
    const li = document.createElement("li");
    li.dataset.state = t.state;
    const label = { pending: "", queued: "queued", running: "moving…", done: `done · ${t.realS.toFixed(0)} s real` }[t.state];
    li.innerHTML = `<span class="q-dot"></span><span>#${t.id} ${t.part.k.label} → ${t.to.where}
      <span class="q-sub">(${t.from.x.toFixed(0)}, ${t.from.y.toFixed(0)}) → (${t.to.x.toFixed(0)}, ${t.to.y.toFixed(0)}) mm</span></span>
      <span class="q-state">${label}</span>`;
    if (t.state === "pending") {
      const actions = li.querySelector(".q-state");
      const ok = document.createElement("button");
      ok.className = "btn btn-primary btn-sm";
      ok.textContent = "Approve";
      ok.onclick = () => { t.state = "queued"; renderQueue(); };
      const no = document.createElement("button");
      no.className = "btn btn-ghost btn-sm";
      no.textContent = "✕";
      no.title = "Discard";
      no.style.marginLeft = "4px";
      no.onclick = () => { scene.remove(t.ghost); t.part.busy = false; tasks = tasks.filter((x) => x !== t); renderQueue(); };
      actions.append(ok, no);
    }
    queueEl.appendChild(li);
  });
}

function logTask(t) {
  let len = 0, actions = 0, prev = sim.tcp;
  const rows = t.waypoints.map((w, i) => {
    const d = Math.hypot(w.tcp[0] - prev[0], w.tcp[1] - prev[1], w.tcp[2] - prev[2]);
    prev = w.tcp;
    len += d;
    actions += w.dwell ? Math.round(GRIP_DWELL_S / 0.1) : Math.max(1, Math.ceil(d / MAX_STEP_MM));
    const v = [...w.tcp.map((n) => n.toFixed(0).padStart(4)), "-180", "0", "0", w.grip ? "1" : "0"];
    return `P${i + 1} ${w.name.padEnd(9)}[${v.join(",")}]`;
  });
  $("tLog").textContent = `# move #${t.id}: ${t.part.k.label} → ${t.to.where}\n# [x,y,z,w,p,r,grip] · 10 Hz · ≤${MAX_STEP_MM} mm/action\n${rows.join("\n")}\n# ${len.toFixed(0)} mm path · ${actions} actions · ≈${(actions * 0.1).toFixed(0)} s at 20 mm/s`;
  $("tCmdInfo").textContent = `move #${t.id}`;
}

// ───────── Motion runner ─────────
const sim = { tcp: [...HOME], gap: 30, task: null, seg: 0, segStart: null, segT: 0, dwellT: 0 };

function stepSim(dt) {
  if (!sim.task) {
    const next = tasks.find((t) => t.state === "queued");
    if (next) {
      sim.task = next; sim.seg = 0; sim.segStart = [...sim.tcp]; sim.segT = 0;
      next.state = "running"; next.realS = 0;
      logTask(next);
      renderQueue();
    } else {
      const d = Math.hypot(...HOME.map((v, i) => v - sim.tcp[i]));
      if (d > 0.5) moveToward(HOME, dt);
      $("tState").textContent = d > 0.5 ? "returning home" : "idle";
      return;
    }
  }
  const t = sim.task;
  const w = t.waypoints[sim.seg];
  t.realS += dt;
  $("tState").textContent = `#${t.id} ${w.name}`;
  if (w.dwell) {
    sim.dwellT += dt;
    const target = w.grip ? t.part.k.grip : 30;
    sim.gap += (target - sim.gap) * Math.min(1, dt * 8);
    if (sim.dwellT >= GRIP_DWELL_S) {
      sim.dwellT = 0;
      sim.gap = target;
      if (w.grip) { t.part.carried = true; }
      else { finishPlace(t); }
      advance();
    }
  } else if (moveToward(w.tcp, dt)) advance();

  if (t.part.carried) {
    t.part.x = sim.tcp[0]; t.part.y = sim.tcp[1];
    placeMesh(t.part, sim.tcp[2] - t.part.k.h / 2);
  }
  $("tGrip").textContent = sim.gap < 29 ? "closed" : "open";
}

function moveToward(target, dt) {
  const d = target.map((v, i) => v - sim.tcp[i]);
  const dist = Math.hypot(...d);
  const step = SPEED_MM_S * dt;
  if (dist <= step) { sim.tcp = [...target]; return true; }
  sim.tcp = sim.tcp.map((v, i) => v + (d[i] / dist) * step);
  return false;
}
function advance() {
  sim.seg++;
  if (sim.seg >= sim.task.waypoints.length) {
    sim.task.state = "done";
    sim.task.part.busy = false;
    status(`Done: ${sim.task.part.k.label} placed in ${sim.task.to.where}. ${sim.task.realS.toFixed(0)} s at real speed.`);
    sim.task = null;
    renderQueue();
  }
}
function finishPlace(t) {
  const p = t.part;
  p.carried = false;
  Object.assign(p, { x: t.to.x, y: t.to.y, z: t.to.z, bin: t.to.bin });
  placeMesh(p);
  scene.remove(t.ghost);
  ghosts.delete(t.id);
}

// ───────── Controls ─────────
function status(msg) { $("tStatus").textContent = msg; }

document.querySelectorAll("[data-speed]").forEach((b) => b.addEventListener("click", () => {
  simSpeed = +b.dataset.speed;
  document.querySelectorAll("[data-speed]").forEach((x) => x.classList.toggle("is-active", x === b));
}));
$("tAuto").addEventListener("change", (e) => {
  if (e.target.checked) { tasks.forEach((t) => { if (t.state === "pending") t.state = "queued"; }); renderQueue(); }
});
$("tReset").addEventListener("click", () => {
  if (!ready) return;
  tasks.forEach((t) => scene.remove(t.ghost));
  tasks = []; ghosts.clear(); taskSeq = 0;
  Object.assign(sim, { tcp: [...HOME], gap: 30, task: null, seg: 0, dwellT: 0 });
  buildScene();
  renderQueue();
  $("tLog").textContent = "// waypoints for the active move appear here";
  $("tCmdInfo").textContent = "cell frame · mm";
  status("Scene reset. Drag a red cube into the yellow bin to start.");
});
$("tExport").addEventListener("click", () => {
  const plan = {
    generated_by: "a-team/drag-to-move (demo mode)",
    status: "pending_human_approval",
    robot: "FANUC ER-4iA",
    frame: "cell: origin robot base on table top, +X forward, +Y left, +Z up (mm)",
    action_format: ["x_mm", "y_mm", "z_mm", "w_deg", "p_deg", "r_deg", "gripper"],
    motion_limits: { sample_period_s: 0.1, maximum_translation_per_action_mm: MAX_STEP_MM, speed_mm_s: SPEED_MM_S },
    moves: tasks.map((t) => ({
      id: t.id, part: t.part.k.label, destination: t.to.where, state: t.state,
      waypoints: t.waypoints.map((w) => ({ name: w.name, tcp: [...w.tcp.map((v) => +v.toFixed(1)), -180, 0, 0, w.grip] })),
    })),
  };
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(plan, null, 2)], { type: "application/json" }));
  a.download = "drag-to-move.plan.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

// ───────── Loop ─────────
function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(host);
resize();

let visible = true;
new IntersectionObserver(([e]) => (visible = e.isIntersecting)).observe(host);
const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const dt = Math.min(0.1, clock.getDelta());
  if (!visible) return;
  if (ready) {
    stepSim(dt * simSpeed);
    poseRobot(sim.tcp, sim.gap);
    const pulse = 0.55 + 0.35 * Math.sin(performance.now() / 260);
    ghosts.forEach((g) => (g.userData.fill.material.opacity = 0.18 + 0.15 * pulse));
  }
  controls.update();
  renderer.render(scene, camera);
});

poseRobot(sim.tcp, sim.gap);
loadCad()
  .then(() => { buildScene(); ready = true; $("twinLoading").remove(); })
  .catch((err) => { console.error(err); $("twinLoading").textContent = "Couldn't load the CAD models."; });
