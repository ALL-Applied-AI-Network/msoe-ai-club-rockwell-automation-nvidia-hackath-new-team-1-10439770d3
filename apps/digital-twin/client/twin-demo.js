/*
 * Demo 3 — "Drag to Move": a live replica of the FANUC ER-4iA cell.
 *
 * The server owns the cell state (twin-api.js). This page renders it and posts
 * user moves; agents report real part and robot poses over the same API, so
 * the replica can mirror the physical cell. Parts are the team's measured CAD
 * (assets/cad/real/*.glb, converted from STEP). The arm is modelled from the
 * ER-4iA link lengths and driven by the shared IK in twin-core.mjs.
 *
 * Cell frame: origin at the robot base on the table top, +X forward, +Y left,
 * +Z up, millimetres. three.js is Y-up, so (x, y, z)cell → (x, z, −y)three.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RGBELoader } from "three/addons/loaders/RGBELoader.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { GTAOPass } from "three/addons/postprocessing/GTAOPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import * as core from "./twin-core.mjs";

const { KINDS, BINS, TOOL, L1, L2, SHOULDER_H, TABLE_X, TABLE_Y, GRIP_OPEN } = core;
const $ = (id) => document.getElementById(id);
const host = $("twin");
const toThree = (x, y, z) => new THREE.Vector3(x, z, -y);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
};

// ───────── Renderer / scene ─────────
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.95;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
host.appendChild(renderer.domElement);

// The viewport is a dark studio in both page themes, like a CAD/sim window.
const BG = new THREE.Color(0x161b22);
const scene = new THREE.Scene();
scene.background = BG;
scene.fog = new THREE.Fog(BG, 2600, 6400);
const overlay = new THREE.Scene(); // ghosts: drawn after AO so they never darken

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(renderer), 0.04).texture;
new RGBELoader().loadAsync("assets/env/studio.hdr").then((tex) => {
  tex.mapping = THREE.EquirectangularReflectionMapping;
  scene.environment = pmrem.fromEquirectangular(tex).texture;
  tex.dispose();
  // Keep the studio HDRI as fill so the key light still shapes the scene.
  scene.traverse((o) => { if (o.material && "envMapIntensity" in o.material) o.material.envMapIntensity = o.material.metalness > 0.5 ? 1 : 0.55; });
}).catch(() => {});

const camera = new THREE.PerspectiveCamera(34, 1, 4, 14000);
const VIEWS = {
  overview: { pos: [1650, -1250, 1150], tgt: [230, 0, 120] },
  work: { pos: [1000, -560, 520], tgt: [330, -10, 110] },
};
camera.position.copy(toThree(...VIEWS.work.pos));
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(toThree(...VIEWS.work.tgt));
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.49;
controls.minDistance = 90;
controls.maxDistance = 4200;

scene.add(new THREE.HemisphereLight(0xeef4ff, 0x30302c, 0.15));
const key = new THREE.DirectionalLight(0xfff3e2, 3.2);
key.position.copy(toThree(950, -760, 1250));
key.target.position.copy(toThree(330, 0, 0));
key.castShadow = true;
key.shadow.mapSize.set(4096, 4096);
Object.assign(key.shadow.camera, { left: -900, right: 900, top: 900, bottom: -900, near: 600, far: 3200 });
key.shadow.bias = -0.00025;
key.shadow.normalBias = 0.6;
key.shadow.radius = 3;
scene.add(key, key.target);
const rim = new THREE.DirectionalLight(0xcfe0ff, 0.9);
rim.position.copy(toThree(-600, 800, 900));
scene.add(rim);

// ───────── Procedural textures ─────────
function canvasTex(size, draw, repeat = 1, srgb = true) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  draw(c.getContext("2d"), size);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(repeat, repeat);
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
function speckle(ctx, s, base, spread, dots, alpha = 0.5) {
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, s, s);
  for (let i = 0; i < dots; i++) {
    const v = Math.random() * spread - spread / 2;
    ctx.fillStyle = `rgba(${v > 0 ? "255,255,255" : "0,0,0"},${Math.abs(v) * alpha})`;
    ctx.fillRect(Math.random() * s, Math.random() * s, 1 + Math.random() * 2, 1 + Math.random() * 2);
  }
}
const tableMap = canvasTex(512, (c, s) => speckle(c, s, "#d3d6da", 0.16, 22000), 6);
const tableRough = canvasTex(256, (c, s) => speckle(c, s, "#9a9a9a", 0.5, 9000, 0.6), 10, false);
const floorMap = canvasTex(1024, (c, s) => {
  speckle(c, s, "#3d4148", 0.22, 90000);
  for (let i = 0; i < 40; i++) { // trowel marks
    c.fillStyle = `rgba(0,0,0,${Math.random() * 0.04})`;
    c.beginPath(); c.ellipse(Math.random() * s, Math.random() * s, 60 + Math.random() * 160, 20 + Math.random() * 50, Math.random() * 3, 0, 7); c.fill();
  }
}, 7);

// ───────── Materials ─────────
const MAT = {
  paint: new THREE.MeshPhysicalMaterial({ color: 0xf5b800, roughness: 0.4, clearcoat: 0.75, clearcoatRoughness: 0.16 }),
  black: new THREE.MeshPhysicalMaterial({ color: 0x17191c, roughness: 0.55, metalness: 0.15, clearcoat: 0.25, clearcoatRoughness: 0.5 }),
  rubber: new THREE.MeshStandardMaterial({ color: 0x101113, roughness: 0.85 }),
  alu: new THREE.MeshPhysicalMaterial({ color: 0xc9ced5, metalness: 1, roughness: 0.3 }),
  aluBrushed: new THREE.MeshPhysicalMaterial({ color: 0xb8bec6, metalness: 1, roughness: 0.38 }),
  anod: new THREE.MeshPhysicalMaterial({ color: 0x2c3036, metalness: 0.75, roughness: 0.34 }),
  steel: new THREE.MeshPhysicalMaterial({ color: 0x8e949b, metalness: 1, roughness: 0.25 }),
  table: new THREE.MeshPhysicalMaterial({ map: tableMap, roughnessMap: tableRough, roughness: 0.7, clearcoat: 0.12, clearcoatRoughness: 0.6 }),
  tape: new THREE.MeshPhysicalMaterial({ color: 0xf2c200, roughness: 0.42, clearcoat: 0.4, clearcoatRoughness: 0.3 }),
  floor: new THREE.MeshStandardMaterial({ map: floorMap, roughness: 0.92 }),
  bin: new THREE.MeshPhysicalMaterial({ color: 0xe36a1c, roughness: 0.5, clearcoat: 0.18, clearcoatRoughness: 0.45 }),
};
function partMaterial(k, color) {
  if (k.material === "blackoxide") return new THREE.MeshPhysicalMaterial({ color: 0x1b1d21, metalness: 0.85, roughness: 0.36, clearcoat: 0.3, clearcoatRoughness: 0.3 });
  if (k.material === "zinc") return new THREE.MeshPhysicalMaterial({ color: 0xcfd3d9, metalness: 1, roughness: 0.2 });
  return new THREE.MeshPhysicalMaterial({ color: color || k.color, roughness: 0.48, clearcoat: 0.2, clearcoatRoughness: 0.45 });
}

function mesh(geo, mat, x = 0, y = 0, z = 0) {
  const m = new THREE.Mesh(geo, mat);
  m.position.set(x, y, z);
  m.castShadow = m.receiveShadow = true;
  return m;
}
const xCyl = (r, h, seg = 48) => new THREE.CylinderGeometry(r, r, h, seg).rotateZ(Math.PI / 2);
const lathe = (pts, seg = 72) => new THREE.LatheGeometry(pts.map(([r, y]) => new THREE.Vector2(r, y)), seg);
// Chamfered motor/joint cap along local Z.
const capZ = (r, h, c = 4) => lathe([[0, -h / 2], [r - c, -h / 2], [r, -h / 2 + c], [r, h / 2 - c], [r - c, h / 2], [0, h / 2]]).rotateX(Math.PI / 2);

// ───────── Floor and table ─────────
const TABLE_T = 24, FLOOR_Z = -780;
{
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(14000, 14000), MAT.floor);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = FLOOR_Z;
  floor.receiveShadow = true;
  scene.add(floor);

  const w = TABLE_X[1] - TABLE_X[0], d = TABLE_Y[1] - TABLE_Y[0];
  const cx = (TABLE_X[0] + TABLE_X[1]) / 2, cy = (TABLE_Y[0] + TABLE_Y[1]) / 2;
  const top = mesh(new RoundedBoxGeometry(w, TABLE_T, d, 4, 3), MAT.table);
  top.position.copy(toThree(cx, cy, -TABLE_T / 2));
  scene.add(top);

  // 45 mm T-slot aluminium frame and legs.
  const h = 22.5, sw = 4.5, sd = 7;
  const prof = new THREE.Shape([
    [-h, -h], [-sw, -h], [-sw, -h + sd], [sw, -h + sd], [sw, -h], [h, -h], [h, -sw], [h - sd, -sw], [h - sd, sw], [h, sw],
    [h, h], [sw, h], [sw, h - sd], [-sw, h - sd], [-sw, h], [-h, h], [-h, sw], [-h + sd, sw], [-h + sd, -sw], [-h, -sw],
  ].map(([a, b]) => new THREE.Vector2(a, b)));
  prof.holes.push(new THREE.Path().absarc(0, 0, 4.2, 0, Math.PI * 2, true));
  const beam = (len) => new THREE.ExtrudeGeometry(prof, { depth: len, bevelEnabled: false, curveSegments: 12 }).translate(0, 0, -len / 2);
  const inset = 40, z0 = -TABLE_T - h;
  const legH = -FLOOR_Z - TABLE_T - 20;
  const bx = [TABLE_X[0] + inset, TABLE_X[1] - inset], by = [TABLE_Y[0] + inset, TABLE_Y[1] - inset];
  for (const y of by) { const m = mesh(beam(bx[1] - bx[0]), MAT.aluBrushed); m.rotation.y = Math.PI / 2; m.position.copy(toThree(cx, y, z0)); scene.add(m); }
  for (const x of bx) { const m = mesh(beam(by[1] - by[0]), MAT.aluBrushed); m.position.copy(toThree(x, cy, z0)); scene.add(m); }
  for (const x of bx) for (const y of by) {
    const leg = mesh(beam(legH), MAT.aluBrushed);
    leg.rotation.x = Math.PI / 2;
    leg.position.copy(toThree(x, y, -TABLE_T - legH / 2));
    const foot = mesh(lathe([[0, 0], [30, 0], [30, 6], [12, 10], [8, 20], [0, 20]], 32), MAT.rubber);
    foot.position.copy(toThree(x, y, FLOOR_Z));
    scene.add(leg, foot);
  }
  // Lower shelf rails.
  for (const y of by) { const m = mesh(beam(bx[1] - bx[0]), MAT.aluBrushed); m.rotation.y = Math.PI / 2; m.position.copy(toThree(cx, y, FLOOR_Z + 160)); scene.add(m); }

  // Yellow floor-marking tape on the front and side edges, as in the real cell.
  const band = (bcx, bcy, sx, sy) => {
    const m = mesh(new RoundedBoxGeometry(sx, 0.5, sy, 1, 0.2), MAT.tape);
    m.position.copy(toThree(bcx, bcy, 0.26));
    m.castShadow = false;
    scene.add(m);
  };
  band(TABLE_X[1] - 26, cy, 48, d - 4);
  band(cx, TABLE_Y[0] + 26, w - 4, 48);
  band(cx, TABLE_Y[1] - 26, w - 4, 48);

  const ring = new THREE.Mesh(new THREE.RingGeometry(503, 505.5, 160, 1, -Math.PI * 0.42, Math.PI * 0.84), new THREE.MeshBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.35, depthWrite: false }));
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.7;
  scene.add(ring);
}

// ───────── Robot (ER-4iA proportions) ─────────
// Link outline: two circles joined by their outer tangents, extruded with a bevel.
function linkGeo(len, r1, r2, depth, bevel = 7) {
  const al = Math.acos((r1 - r2) / len);
  const s = new THREE.Shape();
  s.moveTo(r1 * Math.cos(al), r1 * Math.sin(al));
  s.lineTo(len + r2 * Math.cos(al), r2 * Math.sin(al));
  s.absarc(len, 0, r2, al, -al, true);
  s.lineTo(r1 * Math.cos(-al), r1 * Math.sin(-al));
  s.absarc(0, 0, r1, -al, al - Math.PI * 2, true);
  return new THREE.ExtrudeGeometry(s, { depth: depth - 2 * bevel, bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel, bevelSegments: 5, curveSegments: 48 })
    .translate(0, 0, -(depth - 2 * bevel) / 2);
}

const robot = new THREE.Group();
scene.add(robot);
robot.add(mesh(lathe([[0, 0], [118, 0], [118, 9], [112, 15], [104, 17], [100, 58], [94, 70], [0, 70]]), MAT.paint));
for (let i = 0; i < 8; i++) { // base anchor bolts
  const a = (i / 8) * Math.PI * 2 + Math.PI / 8;
  robot.add(mesh(new THREE.CylinderGeometry(6.5, 6.5, 6, 6), MAT.steel, Math.cos(a) * 110, 18, Math.sin(a) * 110));
}
const ARM_Z = 96; // the upper arm runs beside the turret, as on the real arm
const j1 = new THREE.Group();
j1.position.y = 70;
robot.add(j1);
j1.add(mesh(lathe([[0, 0], [92, 0], [92, 5], [88, 9], [86, 92], [80, 104], [0, 104]]), MAT.paint));
j1.add(mesh(new RoundedBoxGeometry(150, 205, 128, 6, 34), MAT.paint, 0, 190, 0));
j1.add(mesh(capZ(30, 80), MAT.black, -88, 130, 0).rotateY(Math.PI / 2)); // J1 motor
const j2 = new THREE.Group();
j2.position.y = SHOULDER_H - 70;
j1.add(j2);
j2.add(mesh(capZ(58, 34), MAT.black, 0, 0, 52));
j2.add(mesh(capZ(66, 66), MAT.paint, 0, 0, ARM_Z));
j2.add(mesh(capZ(48, 46), MAT.black, 0, 0, ARM_Z + 52));
j2.add(mesh(capZ(20, 8), MAT.alu, 0, 0, ARM_Z + 78));
j2.add(mesh(linkGeo(L1, 60, 46, 62), MAT.paint, 0, 0, ARM_Z));
const j3 = new THREE.Group();
j3.position.x = L1;
j2.add(j3);
j3.add(mesh(capZ(50, 64), MAT.paint, 0, 0, ARM_Z));
j3.add(mesh(capZ(46, 96), MAT.paint, 0, 0, ARM_Z / 2 - 6));
j3.add(mesh(capZ(36, 24), MAT.black, 0, 0, ARM_Z + 44));
j3.add(mesh(linkGeo(L2, 44, 30, 54, 6), MAT.paint));
j3.add(mesh(new RoundedBoxGeometry(96, 70, 64, 5, 16), MAT.black, -64, 0, 0)); // J4 motor housing
const wrist = new THREE.Group();
wrist.position.x = L2;
j3.add(wrist);
wrist.add(mesh(capZ(32, 60), MAT.paint));
wrist.add(mesh(capZ(25, 66, 3), MAT.black));
wrist.add(mesh(xCyl(24, 8), MAT.alu, 32, 0, 0)); // J6 flange
const flangeBolts = new THREE.Group();
for (let i = 0; i < 4; i++) {
  const a = (i / 4) * Math.PI * 2 + Math.PI / 4;
  flangeBolts.add(mesh(xCyl(2.6, 3, 12), MAT.anod, 37, Math.cos(a) * 16, Math.sin(a) * 16));
}
wrist.add(flangeBolts);
wrist.add(mesh(xCyl(28, 6), MAT.anod, 39, 0, 0));
const FINGER_BASE = TOOL + 2 - 76.2; // fingertips 2 mm past the TCP
wrist.add(mesh(new RoundedBoxGeometry(FINGER_BASE - 42, 34, 60, 3, 3).translate((FINGER_BASE + 42) / 2, 0, 0), MAT.anod));
wrist.add(mesh(new THREE.BoxGeometry(2, 18, 62).translate(FINGER_BASE - 1, 0, 0), MAT.alu));
const fingers = new THREE.Group();
wrist.add(fingers);
let fingerA = null, fingerB = null;

// Dress pack: a cable from the base to the elbow, re-swept every pose.
const anchors = [
  [robot, -112, 40, 0], [j1, -84, 205, 34], [j2, 30, 50, ARM_Z + 46], [j2, L1 - 70, 44, ARM_Z + 46], [j3, -40, 34, 40],
].map(([parent, x, y, z]) => { const o = new THREE.Object3D(); o.position.set(x, y, z); parent.add(o); return o; });
const cable = mesh(new THREE.BufferGeometry(), MAT.rubber);
scene.add(cable);
function sweepCable() {
  robot.updateMatrixWorld(true);
  const pts = anchors.map((o) => o.getWorldPosition(new THREE.Vector3()));
  cable.geometry.dispose();
  cable.geometry = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts, false, "centripetal"), 64, 8, 10, false);
}

function poseRobot(tcp, gap) {
  const { j1: t1, a, b } = core.solveIK(tcp);
  j1.rotation.y = t1;
  j2.rotation.z = a;
  j3.rotation.z = b - a;
  wrist.rotation.z = -Math.PI / 2 - b;
  if (fingerA) { fingerA.position.z = gap / 2; fingerB.position.z = -gap / 2; }
  sweepCable();
}

// ───────── CAD ─────────
const gltf = new GLTFLoader();
const geos = {};
function grounded(geo) {
  const g = geo.clone();
  g.computeBoundingBox();
  const b = g.boundingBox;
  g.translate(-(b.min.x + b.max.x) / 2, -b.min.y, -(b.min.z + b.max.z) / 2);
  g.computeBoundingSphere();
  return g;
}
async function loadCad() {
  const files = [...new Set([...Object.values(KINDS).map((k) => k.file), "box_orange.glb", "finger_left.glb", "finger_right.glb"])];
  await Promise.all(files.map(async (f) => {
    const g = await gltf.loadAsync(`assets/cad/real/${f}`);
    let geo = null;
    g.scene.traverse((o) => { if (o.isMesh && !geo) geo = o.geometry; });
    geos[f] = geo;
  }));
  // CAD fingers: length along +Z, grip face at +Y = 13, width centred at X = 4.
  const fingerGeo = (f, basis) => geos[f].clone().translate(-4, -13, 0).applyMatrix4(new THREE.Matrix4().makeBasis(...basis));
  fingerA = mesh(fingerGeo("finger_left.glb", [new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, -1), new THREE.Vector3(1, 0, 0)]), MAT.aluBrushed, FINGER_BASE);
  fingerB = mesh(fingerGeo("finger_right.glb", [new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(1, 0, 0)]), MAT.aluBrushed, FINGER_BASE);
  fingers.add(fingerA, fingerB);
  for (const k of Object.values(KINDS)) k.geo = grounded(geos[k.file]);
  // Bins are ordinary stateful part views. This lets live observations move them
  // and lets the simulator carry their contents when a user drags a loaded bin.
}

// ───────── Live state ─────────
const S = { parts: {}, moves: [], robot: null, mode: "sim", sim: { speed: 4, autoApprove: true }, agent: {}, clients: 0 };
const views = new Map(); // part id → { mesh, grab }
const ghosts = new Map(); // move id → ghost group
const OPEN = new Set(["pending", "approved", "running"]);
const renderTcp = [...core.HOME];
let renderGap = GRIP_OPEN;
let ready = false, connected = false;

function partView(p) {
  let v = views.get(p.id);
  const k = KINDS[p.kind];
  if (v && v.kind !== p.kind) { scene.remove(v.mesh); views.delete(p.id); v = null; }
  if (!v) {
    const m = mesh(k.geo, partMaterial(k, p.color));
    const grab = new THREE.Mesh(new THREE.SphereGeometry(k.r + 10, 12, 8), new THREE.MeshBasicMaterial({ visible: false }));
    grab.position.y = k.h / 2;
    grab.userData.id = p.id;
    m.add(grab);
    m.position.copy(toThree(p.x, p.y, p.z));
    scene.add(m);
    v = { mesh: m, grab, kind: p.kind, target: new THREE.Vector3() };
    views.set(p.id, v);
  } else if (p.color && v.mesh.material.color.getHexString() !== p.color.slice(1)) v.mesh.material.color.set(p.color);
  v.target.copy(toThree(p.x, p.y, p.z));
  v.mesh.rotation.y = THREE.MathUtils.degToRad(p.yaw || 0);
  return v;
}
function dropView(id) {
  const v = views.get(id);
  if (v) { scene.remove(v.mesh); v.mesh.material.dispose(); views.delete(id); }
}

function makeGhost(kind, pending) {
  const k = KINDS[kind];
  const g = new THREE.Group();
  const fill = new THREE.Mesh(k.geo, new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.3, depthWrite: false, depthTest: false }));
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(k.geo, 28), new THREE.LineBasicMaterial({ transparent: true, opacity: 0.95, depthTest: false }));
  const marker = new THREE.Mesh(new THREE.RingGeometry(k.r + 4, k.r + 6.5, 48), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.85, side: THREE.DoubleSide, depthTest: false }));
  marker.rotation.x = -Math.PI / 2;
  marker.position.y = 0.9;
  g.add(fill, edges, marker);
  g.userData = { fill, edges, marker };
  tintGhost(g, pending ? "pending" : "ok");
  overlay.add(g);
  return g;
}
function tintGhost(g, state) {
  const c = { ok: 0x22d3ee, pending: 0xf5a524, bad: 0xf36b6b }[state];
  for (const m of Object.values(g.userData)) m.material.color.setHex(c);
}
function syncGhosts() {
  if (!ready) return;
  const open = new Map(S.moves.filter((m) => OPEN.has(m.status) && S.parts[m.partId]).map((m) => [m.id, m]));
  for (const [id, g] of ghosts) if (!open.has(id)) { overlay.remove(g); ghosts.delete(id); }
  for (const [id, m] of open) {
    let g = ghosts.get(id);
    if (!g) { g = makeGhost(m.kind, m.status === "pending"); ghosts.set(id, g); }
    tintGhost(g, m.status === "pending" ? "pending" : "ok");
    g.position.copy(toThree(m.to.x, m.to.y, m.to.z));
  }
}
const busy = (id) => S.parts[id]?.carried || S.moves.some((m) => m.partId === id && OPEN.has(m.status));

// ───────── Server link ─────────
async function api(method, path, body) {
  const r = await fetch(`/api/${path}`, {
    method,
    credentials: "same-origin",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(j.error || r.statusText), { status: r.status });
  return j;
}

const prevStatus = new Map();
function onMove(m) {
  const i = S.moves.findIndex((x) => x.id === m.id);
  if (i >= 0) S.moves[i] = m; else S.moves.push(m);
  const before = prevStatus.get(m.id);
  if (before !== m.status) {
    prevStatus.set(m.id, m.status);
    if (m.status === "done") status(`Done: ${m.label} → ${m.to.where}${m.note ? ` (${m.note})` : m.elapsedS ? ` · ${m.elapsedS.toFixed(0)} s at real speed` : ""}.`);
    if (m.status === "running" && S.mode === "live") status(`Agent is executing #${m.id}: ${m.label} → ${m.to.where}.`);
    if (m.status === "failed") status(`Move #${m.id} failed${m.note ? `: ${m.note}` : ""}.`);
  }
  if (m.status === "running") logMove(m);
}
function applySnapshot(s) {
  S.mode = s.mode; S.sim = s.sim; S.agent = s.agent || {}; S.clients = s.clients;
  S.moves = s.moves;
  for (const m of S.moves) prevStatus.set(m.id, m.status);
  S.parts = Object.fromEntries(s.parts.map((p) => [p.id, p]));
  for (const id of [...views.keys()]) if (!S.parts[id]) dropView(id);
  if (ready) for (const p of s.parts) partView(p);
  S.robot = s.robot;
  renderHud();
  syncGhosts();
  renderControls();
  renderQueue();
}

function connect() {
  const es = new EventSource("/api/events");
  const on = (ev, fn) => es.addEventListener(ev, (e) => { fn(JSON.parse(e.data)); });
  on("snapshot", (d) => { connected = true; applySnapshot(d); renderLink(); });
  on("part", ({ part }) => { S.parts[part.id] = part; if (ready) partView(part); });
  on("part_removed", ({ id }) => { delete S.parts[id]; dropView(id); syncGhosts(); });
  on("move", ({ move }) => { onMove(move); syncGhosts(); renderQueue(); });
  on("robot", ({ robot }) => { S.robot = robot; renderHud(); });
  on("sim", ({ sim }) => { S.sim = sim; renderControls(); });
  on("agent", ({ agent }) => { S.agent = agent; renderLink(); });
  es.onopen = () => { connected = true; renderLink(); };
  es.onerror = () => { connected = false; renderLink(); };
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
  const grabs = [...views.entries()].filter(([id]) => !busy(id)).map(([, v]) => v.grab);
  return ray.intersectObjects(grabs, false)[0]?.object.userData.id || null;
}
function tablePoint() {
  const v = new THREE.Vector3();
  return ray.ray.intersectPlane(tablePlane, v) ? { x: v.x, y: -v.z } : null;
}
function glow(id, on) {
  const v = views.get(id);
  if (!v) return;
  v.mesh.material.emissive.setHex(0x22d3ee);
  v.mesh.material.emissiveIntensity = on ? 0.35 : 0;
}

renderer.domElement.addEventListener("pointerdown", (e) => {
  if (!ready || !connected) return;
  setPointer(e);
  const id = hitPart();
  if (!id) return;
  const p = S.parts[id];
  const pt = tablePoint();
  controls.enabled = false;
  renderer.domElement.setPointerCapture(e.pointerId);
  drag = { id, off: pt ? { x: p.x - pt.x, y: p.y - pt.y } : { x: 0, y: 0 }, ghost: makeGhost(p.kind, false), to: null };
  host.classList.add("is-dragging");
  glow(id, true);
  updateDrag();
});
renderer.domElement.addEventListener("pointermove", (e) => {
  setPointer(e);
  if (drag) return updateDrag();
  if (ready) {
    const id = hitPart();
    host.classList.toggle("is-hover", !!id);
    if (hover !== id) { glow(hover, false); glow(id, true); hover = id; }
  }
});
let hover = null;
renderer.domElement.addEventListener("pointerup", async (e) => {
  if (!drag) return;
  renderer.domElement.releasePointerCapture(e.pointerId);
  controls.enabled = true;
  host.classList.remove("is-dragging");
  const { id, ghost, to } = drag;
  glow(id, false);
  drag = null;
  const p = S.parts[id];
  if (!to || !p || Math.hypot(to.x - p.x, to.y - p.y) < 8) { overlay.remove(ghost); return; }
  if (!to.ok) { overlay.remove(ghost); status(`Can't place ${KINDS[p.kind].label.toLowerCase()} there: ${to.why}.`); return; }
  try {
    const m = await api("POST", "moves", { partId: id, x: to.x, y: to.y, bin: to.bin });
    onMove(m);
    syncGhosts();
    renderQueue();
    status(m.status === "pending" ? `Drafted #${m.id}: ${m.label} → ${m.to.where}. Approve it in the queue.`
      : S.mode === "live" ? `Sent #${m.id} to the cell: ${m.label} → ${m.to.where}. The ghost stays until an agent reports it moved.`
      : `Queued #${m.id}: ${m.label} → ${m.to.where}.`);
  } catch (err) {
    status(`Server refused the move: ${err.message}.`);
  }
  overlay.remove(ghost);
});

function updateDrag() {
  const pt = tablePoint();
  if (!pt) return;
  const p = S.parts[drag.id];
  const to = core.resolveDrop(S, p, pt.x + drag.off.x, pt.y + drag.off.y);
  drag.to = to;
  drag.ghost.position.copy(toThree(to.x, to.y, to.z));
  tintGhost(drag.ghost, to.ok ? "ok" : "bad");
  status(to.ok ? `Target: ${to.where} (${to.x.toFixed(0)}, ${to.y.toFixed(0)})` : `✗ ${to.why}`);
}

// ───────── Panels ─────────
function status(msg) { $("tStatus").textContent = msg; }

const queueEl = $("tQueue");
function renderQueue() {
  const list = S.moves.slice(-40);
  $("tCount").textContent = `${S.moves.filter((m) => OPEN.has(m.status)).length} open`;
  $("tExport").disabled = !S.moves.length;
  if (!list.length) { queueEl.innerHTML = '<li class="empty">No moves yet. Drag a part in the replica.</li>'; return; }
  queueEl.innerHTML = "";
  for (const m of [...list].reverse()) {
    const li = document.createElement("li");
    li.dataset.state = m.status === "approved" ? "queued" : m.status === "done" ? "done" : m.status;
    const label = {
      pending: "",
      approved: S.mode === "live" ? "waiting for agent" : "queued",
      running: m.phase ? `${m.phase}…` : "moving…",
      done: m.elapsedS ? `done · ${m.elapsedS.toFixed(0)} s` : "done",
      failed: "failed",
      rejected: "rejected",
      cancelled: "cancelled",
    }[m.status];
    li.innerHTML = `<span class="q-dot"></span><span>#${m.id} ${m.label} → ${m.to.where}${m.origin === "agent" ? ' <span class="q-tag">agent</span>' : ""}
      <span class="q-sub">(${m.from.x.toFixed(0)}, ${m.from.y.toFixed(0)}) → (${m.to.x.toFixed(0)}, ${m.to.y.toFixed(0)}) mm</span></span>
      <span class="q-state">${label}</span>`;
    const actions = li.querySelector(".q-state");
    const button = (text, cls, title, fn) => {
      const b = document.createElement("button");
      b.className = `btn ${cls} btn-sm`;
      b.type = "button";
      b.textContent = text;
      b.title = title;
      b.onclick = () => fn().catch((err) => status(err.message));
      actions.append(b);
    };
    if (m.status === "pending") button("Approve", "btn-primary", "Approve this move", () => api("PATCH", `moves/${m.id}`, { status: "approved" }));
    if (m.status === "pending" || (m.status === "approved" && S.mode === "live"))
      button("✕", "btn-ghost", "Cancel this move", () => api("PATCH", `moves/${m.id}`, { status: "cancelled" }));
    queueEl.appendChild(li);
  }
}

function logMove(m) {
  const rows = m.waypoints.map((w, i) => {
    const v = [...w.tcp.map((n) => n.toFixed(0).padStart(4)), "-180", "0", "0", w.grip ? "1" : "0"];
    return `P${i + 1} ${w.name.padEnd(9)}[${v.join(",")}]`;
  });
  const st = m.stats || { len: 0, actions: 0 };
  $("tLog").textContent = `# move #${m.id}: ${m.label} → ${m.to.where}\n# [x,y,z,w,p,r,grip] · 10 Hz · ≤${core.MAX_STEP_MM} mm/action\n${rows.join("\n")}\n# ${st.len.toFixed(0)} mm path · ${st.actions} actions · ≈${(st.actions * 0.1).toFixed(0)} s at 20 mm/s`;
  $("tCmdInfo").textContent = `move #${m.id}`;
}

const jointEls = [...document.querySelectorAll("#tJoints b")];
function renderHud() {
  const r = S.robot;
  if (!r) return;
  $("tTcp").textContent = r.tcp.slice(0, 3).map((v) => v.toFixed(0)).join(", ");
  $("tGrip").textContent = r.gripper >= GRIP_OPEN - 1 ? "open" : `${r.gripper.toFixed(0)} mm`;
  $("tState").textContent = r.state;
  $("tSrc").textContent = r.source === "agent" ? "agent" : r.source === "sim" ? "sim" : r.source;
  (r.joints || []).forEach((v, i) => jointEls[i] && (jointEls[i].textContent = `${v.toFixed(1)}°`));
}

function renderControls() {
  document.querySelectorAll("[data-mode]").forEach((b) => b.classList.toggle("is-active", b.dataset.mode === S.mode));
  document.querySelectorAll("[data-speed]").forEach((b) => {
    b.classList.toggle("is-active", +b.dataset.speed === S.sim.speed);
    b.disabled = S.mode === "live";
  });
  $("tAuto").checked = S.sim.autoApprove;
  $("twinBadge").textContent = S.mode === "live" ? "Live · agents report the real cell" : "Sim · server-side simulation";
  $("twinBadge").classList.toggle("badge-live", S.mode === "live");
  host.dataset.mode = S.mode;
}

function ago(t) {
  if (!t) return "never";
  const s = Math.max(0, (Date.now() - t) / 1000);
  return s < 2 ? "just now" : s < 90 ? `${s.toFixed(0)} s ago` : `${(s / 60).toFixed(0)} min ago`;
}
function renderLink() {
  $("tLink").dataset.on = connected ? "1" : "0";
  $("tLinkText").textContent = connected ? `connected · ${S.mode} mode` : "reconnecting…";
  $("tAgent").textContent = S.agent?.lastSeen ? `${S.agent.name} · ${ago(S.agent.lastSeen)} · ${S.agent.updates} updates` : "no agent yet";
}
setInterval(renderLink, 1000);

document.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () =>
  api("PUT", "mode", { mode: b.dataset.mode })
    .then(() => status(b.dataset.mode === "live" ? "Live mode: agents now drive the arm and parts. Your moves wait for an agent." : "Sim mode: the server animates the arm."))
    .catch((err) => status(err.message))));
document.querySelectorAll("[data-speed]").forEach((b) => b.addEventListener("click", () => api("PUT", "sim", { speed: +b.dataset.speed }).catch((err) => status(err.message))));
$("tAuto").addEventListener("change", (e) => api("PUT", "sim", { autoApprove: e.target.checked }).catch((err) => status(err.message)));
$("tReset").addEventListener("click", () => api("POST", "reset").then(() => {
  $("tLog").textContent = "// waypoints for the active move appear here";
  $("tCmdInfo").textContent = "cell frame · mm";
  status("Scene reset. Drag a cube into an orange bin to start.");
}).catch((err) => status(err.message)));
$("tExport").addEventListener("click", () => {
  const plan = {
    generated_by: "a-team/drag-to-move",
    status: "pending_human_approval",
    robot: "FANUC ER-4iA",
    frame: core.SCHEMA.frame,
    action_format: ["x_mm", "y_mm", "z_mm", "w_deg", "p_deg", "r_deg", "gripper"],
    motion_limits: { sample_period_s: 0.1, maximum_translation_per_action_mm: core.MAX_STEP_MM, speed_mm_s: core.SPEED_MM_S },
    moves: S.moves.map((m) => ({
      id: m.id, part: m.partId, kind: m.kind, destination: m.to.where, state: m.status, origin: m.origin,
      waypoints: m.waypoints.map((w) => ({ name: w.name, tcp: [...w.tcp.map((v) => +v.toFixed(1)), -180, 0, 0, w.grip] })),
    })),
  };
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(plan, null, 2)], { type: "application/json" }));
  a.download = "drag-to-move.plan.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

// Camera presets; "gripper" follows the TCP while keeping the orbit offset.
let follow = false, camAnim = null;
document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("[data-view]").forEach((x) => x.classList.toggle("is-active", x === b));
  follow = b.dataset.view === "gripper";
  if (follow) {
    const t = toThree(renderTcp[0], renderTcp[1], renderTcp[2] - 20);
    camAnim = { p0: camera.position.clone(), t0: controls.target.clone(), p1: t.clone().add(new THREE.Vector3(210, 100, -80)), t1: t, k: 0 };
  } else {
    const v = VIEWS[b.dataset.view];
    camAnim = { p0: camera.position.clone(), t0: controls.target.clone(), p1: toThree(...v.pos), t1: toThree(...v.tgt), k: 0 };
  }
}));

// ───────── Post-processing ─────────
let quality = store.get("twinQuality") || (matchMedia("(pointer: coarse)").matches ? "fast" : "high");
const composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 }));
composer.addPass(new RenderPass(scene, camera));
const gtao = new GTAOPass(scene, camera, 1, 1);
gtao.updateGtaoMaterial({ radius: 70, distanceExponent: 1, thickness: 30, scale: 2.2, samples: 16, distanceFallOff: 1 });
gtao.updatePdMaterial({ lumaPhi: 10, depthPhi: 2, normalPhi: 3, radius: 5, rings: 2, samples: 16 });
gtao.blendIntensity = 0.9;
if (new URLSearchParams(location.search).has("ao")) gtao.output = GTAOPass.OUTPUT.Denoise;
composer.addPass(gtao);
const overlayPass = new RenderPass(overlay, camera);
overlayPass.clear = false;
composer.addPass(overlayPass);
composer.addPass(new OutputPass());
function setQuality(q) {
  quality = q;
  store.set("twinQuality", q);
  $("tQuality").textContent = q === "high" ? "High quality" : "Fast";
  $("tQuality").setAttribute("aria-pressed", q === "high");
  renderer.shadowMap.type = q === "high" ? THREE.PCFSoftShadowMap : THREE.PCFShadowMap;
  key.shadow.mapSize.setScalar(q === "high" ? 4096 : 2048);
  key.shadow.map?.dispose();
  key.shadow.map = null;
}
$("tQuality").addEventListener("click", () => setQuality(quality === "high" ? "fast" : "high"));
setQuality(quality);

function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  composer.setPixelRatio(renderer.getPixelRatio());
  composer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(host);
resize();

// ───────── Loop ─────────
let visible = true;
new IntersectionObserver(([e]) => (visible = e.isIntersecting)).observe(host);
const clock = new THREE.Clock();
const lastPose = [Infinity, Infinity, Infinity, Infinity]; // forces the first pose
renderer.setAnimationLoop(() => {
  const dt = Math.min(0.1, clock.getDelta());
  if (!visible) return;
  const k = 1 - Math.exp(-dt * 14);
  if (S.robot) {
    for (let i = 0; i < 3; i++) renderTcp[i] += (S.robot.tcp[i] - renderTcp[i]) * k;
    renderGap += (S.robot.gripper - renderGap) * k;
  }
  if (Math.abs(lastPose[0] - renderTcp[0]) + Math.abs(lastPose[1] - renderTcp[1]) + Math.abs(lastPose[2] - renderTcp[2]) + Math.abs(lastPose[3] - renderGap) > 0.01) {
    poseRobot(renderTcp, renderGap);
    lastPose.splice(0, 4, ...renderTcp, renderGap);
  }
  if (ready) {
    for (const [id, v] of views) {
      const p = S.parts[id];
      if (p?.carried) v.mesh.position.copy(toThree(renderTcp[0], renderTcp[1], renderTcp[2] - KINDS[p.kind].h / 2));
      else v.mesh.position.lerp(v.target, k);
    }
    const pulse = 0.55 + 0.35 * Math.sin(performance.now() / 260);
    for (const g of ghosts.values()) g.userData.fill.material.opacity = 0.16 + 0.16 * pulse;
  }
  if (camAnim) {
    camAnim.k = Math.min(1, camAnim.k + dt * 1.6);
    const e = 1 - Math.pow(1 - camAnim.k, 3);
    camera.position.lerpVectors(camAnim.p0, camAnim.p1, e);
    controls.target.lerpVectors(camAnim.t0, camAnim.t1, e);
    if (camAnim.k >= 1) camAnim = null;
  } else if (follow) {
    const d = toThree(renderTcp[0], renderTcp[1], renderTcp[2] - 20).sub(controls.target).multiplyScalar(k);
    controls.target.add(d);
    camera.position.add(d);
  }
  controls.update();
  if (quality === "high") composer.render();
  else {
    renderer.render(scene, camera);
    renderer.autoClear = false;
    renderer.render(overlay, camera);
    renderer.autoClear = true;
  }
});

poseRobot(renderTcp, renderGap);
if (new URLSearchParams(location.search).has("debug")) window.__twin = { S, wrist, renderTcp };
connect();
loadCad()
  .then(() => {
    ready = true;
    for (const p of Object.values(S.parts)) partView(p);
    for (const v of views.values()) v.mesh.position.copy(v.target);
    syncGhosts();
    $("twinLoading").remove();
  })
  .catch((err) => { console.error(err); $("twinLoading").textContent = "Couldn't load the CAD models."; });
