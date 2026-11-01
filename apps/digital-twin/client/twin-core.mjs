/*
 * Shared cell model for the Drag-to-Move twin. Loaded by the browser (rendering,
 * drag previews) and by server.js (authoritative state, validation, simulation),
 * so a move is judged by exactly the same rules on both sides.
 *
 * Cell frame: origin at the robot base on the table top, +X forward, +Y left,
 * +Z up, millimetres, degrees.
 */

// ER-4iA kinematics (scene_config.py) and cell limits (robot-mvp/policy.py).
export const SHOULDER_H = 330, L1 = 260, L2 = 290, TOOL = 140;
export const SPEED_MM_S = 20;
export const MAX_STEP_MM = 2;
export const GRIP_DWELL_S = 0.5;
export const SAFE_Z = 140;
export const HOME = [300, 0, 250];
export const GRIP_OPEN = 34;
export const TABLE_X = [-371, 759], TABLE_Y = [-772, 792];
export const LIVE_DONE_TOL_MM = 6;

// Parts measured from the team's STEP files (assets/cad/real/*.glb).
// h: height as it rests, r: footprint radius, grip: finger gap when closed.
export const KINDS = {
  cube_big: { file: "cube_big.glb", label: "Cube 21 mm", h: 21, r: 15, grip: 21, up: "y", color: "#c8372d", material: "plastic" },
  cube_small: { file: "cube_small.glb", label: "Cube 12 mm", h: 12, r: 9, grip: 12, up: "y", color: "#2f6fd6", material: "plastic" },
  cube_yellow: { file: "cube_yellow.glb", label: "Yellow cube 12 mm", h: 12, r: 9, grip: 12, up: "y", color: "#f2c230", material: "plastic" },
  screw_m6: { file: "screw_m6x25.glb", label: "M6×25 cap screw", h: 10, r: 17, grip: 6, up: "side", color: "#1d1f23", material: "blackoxide" },
  screw_pan: { file: "screw_pan.glb", label: "Pan head screw", h: 12, r: 14, grip: 6, up: "side", color: "#c7cbd1", material: "zinc" },
  box_orange: { file: "box_orange.glb", label: "Orange bin", h: 54, r: 36, grip: 8, up: "y", color: "#e86f18", material: "plastic", isBin: true },
  box_blue: { file: "box_orange.glb", label: "Blue bin", h: 54, r: 36, grip: 8, up: "y", color: "#4aa8d8", material: "plastic", isBin: true },
  box_yellow: { file: "box_orange.glb", label: "Yellow bin", h: 54, r: 36, grip: 8, up: "y", color: "#e5b91f", material: "plastic", isBin: true },
};

// Orange box: 50 × 50 outer, 54 tall, open top, 2 mm floor.
export const BIN_HALF = 25, BIN_FLOOR = 2, BIN_H = 54;
export const BINS = [
  { id: "bin_orange", kind: "box_orange", label: "orange bin", x: 223, y: 71, color: "#e86f18" },
  { id: "bin_blue", kind: "box_blue", label: "blue bin", x: 346, y: 226, color: "#4aa8d8" },
  { id: "bin_yellow", kind: "box_yellow", label: "yellow bin", x: 493, y: -22, color: "#e5b91f" },
];

export const DEFAULT_LAYOUT = [
  ["cube_big", 360, -95], ["cube_big", 405, -60],
  ["cube_small", 385, -140], ["cube_small", 430, -110], ["cube_small", 345, -45],
  ["cube_yellow", 320, 45], ["cube_yellow", 300, 95], ["cube_yellow", 350, 80],
  ["screw_m6", 450, 30], ["screw_m6", 480, 70],
  ["screw_pan", 440, 115], ["screw_pan", 478, 150],
];

export function defaultParts() {
  const count = {};
  const bins = BINS.map(({ id, kind, x, y, color }) => ({ id, kind, x, y, z: 0, yaw: 0, bin: null, color }));
  const loose = DEFAULT_LAYOUT.map(([kind, x, y]) => {
    count[kind] = (count[kind] || 0) + 1;
    return { id: `${kind}_${count[kind]}`, kind, x, y, z: 0, yaw: 0, bin: null };
  });
  return bins.concat(loose);
}

// ───────── Kinematics ─────────
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

export function solveIK([x, y, z]) {
  const j1 = Math.atan2(y, x);
  const dx = Math.hypot(x, y), dz = z + TOOL - SHOULDER_H;
  const D = clamp(Math.hypot(dx, dz), Math.abs(L1 - L2) + 1, L1 + L2 - 1);
  const q = Math.acos(clamp((D * D - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1, 1));
  const a = Math.atan2(dz, dx) + Math.atan2(L2 * Math.sin(q), L1 + L2 * Math.cos(q));
  return { j1, a, b: a - q };
}

// FANUC-style readout: J2 from vertical, J3 forearm from horizontal, J5 keeps the tool vertical.
export function jointReadout(tcp) {
  const { j1, a, b } = solveIK(tcp);
  const d = (r) => (r * 180) / Math.PI;
  return [d(j1), 90 - d(a), d(b), 0, -90 - d(b), 0];
}

export function reachable([x, y, z]) {
  const r = Math.hypot(x, y);
  return r > 190 && Math.hypot(r, z + TOOL - SHOULDER_H) < L1 + L2 - 18;
}

// ───────── Placement rules ─────────
const OPEN_MOVES = new Set(["pending", "approved", "running"]);

function spotsTaken(state, partId) {
  const spots = [];
  for (const p of Object.values(state.parts)) {
    if (p.id === partId || p.carried) continue;
    spots.push({ x: p.x, y: p.y, r: KINDS[p.kind].r, bin: p.bin });
  }
  for (const m of state.moves) {
    if (m.partId === partId || !OPEN_MOVES.has(m.status)) continue;
    const p = state.parts[m.partId];
    if (p) spots.push({ x: m.to.x, y: m.to.y, r: KINDS[p.kind].r, bin: m.to.bin });
  }
  return spots;
}

function binsInState(state) {
  return BINS.map((b) => {
    const p = state.parts[b.id];
    return p ? { ...b, x: p.x, y: p.y, z: p.z || 0, yaw: p.yaw || 0 } : b;
  });
}

/** Where would `part` land if dropped at (x, y)? Returns { x, y, z, bin, where, ok, why? }. */
export function resolveDrop(state, part, x, y, binId = null) {
  const k = KINDS[part.kind];
  const bins = binsInState(state);
  const binHit = k.isBin ? null : (binId ? bins.find((b) => b.id === binId) : bins.find((b) => Math.abs(x - b.x) < BIN_HALF + 10 && Math.abs(y - b.y) < BIN_HALF + 10));
  if (binHit) {
    const b = binHit;
    const slots = part.kind.startsWith("screw") ? [[-11, 0], [11, 0], [0, 0]] : k.h > 15 ? [[-12, -12], [12, -12], [-12, 12], [12, 12]] : [[-11, -11], [11, -11], [-11, 11], [11, 11], [0, 0]];
    const taken = spotsTaken(state, part.id).filter((s) => s.bin === b.id);
    const slot = slots.find(([sx, sy]) => taken.every((s) => Math.hypot(s.x - (b.x + sx), s.y - (b.y + sy)) > 10));
    if (!slot) return { x, y, z: BIN_FLOOR, bin: b.id, where: b.label, ok: false, why: `${b.label} is full` };
    return { x: b.x + slot[0], y: b.y + slot[1], z: BIN_FLOOR, bin: b.id, where: b.label, ok: true };
  }
  const to = { x, y, z: 0, bin: null, where: "table", ok: true };
  if (!Number.isFinite(x) || !Number.isFinite(y)) return { ...to, ok: false, why: "invalid coordinates" };
  if (x < TABLE_X[0] + 60 || x > TABLE_X[1] - 60 || y < TABLE_Y[0] + 60 || y > TABLE_Y[1] - 60) return { ...to, ok: false, why: "off the table" };
  if (bins.some((b) => b.id !== part.id && Math.abs(x - b.x) < BIN_HALF + k.r + 4 && Math.abs(y - b.y) < BIN_HALF + k.r + 4)) return { ...to, ok: false, why: "collides with a bin" };
  if (spotsTaken(state, part.id).some((s) => !s.bin && Math.hypot(s.x - x, s.y - y) < s.r + k.r)) return { ...to, ok: false, why: "spot is occupied" };
  if (!reachable([x, y, k.h / 2]) || !reachable([x, y, SAFE_Z])) return { ...to, ok: false, why: "outside the arm's reach" };
  return to;
}

/** Eight-waypoint pick-and-place at the cell's safe height. */
export function planMove(part, to) {
  const k = KINDS[part.kind];
  const pz = part.z + k.h / 2, dz = to.z + k.h / 2;
  return [
    { name: "approach", tcp: [part.x, part.y, SAFE_Z], grip: 0 },
    { name: "descend", tcp: [part.x, part.y, pz], grip: 0 },
    { name: "grasp", tcp: [part.x, part.y, pz], grip: 1, dwell: true },
    { name: "lift", tcp: [part.x, part.y, SAFE_Z], grip: 1 },
    { name: "transfer", tcp: [to.x, to.y, SAFE_Z], grip: 1 },
    { name: "lower", tcp: [to.x, to.y, dz], grip: 1 },
    { name: "release", tcp: [to.x, to.y, dz], grip: 0, dwell: true },
    { name: "retreat", tcp: [to.x, to.y, SAFE_Z], grip: 0 },
  ];
}

/** TCP action count and path length the way the policy would stream it (10 Hz, ≤2 mm/action). */
export function pathStats(start, waypoints) {
  let len = 0, actions = 0, prev = start;
  for (const w of waypoints) {
    const d = Math.hypot(w.tcp[0] - prev[0], w.tcp[1] - prev[1], w.tcp[2] - prev[2]);
    prev = w.tcp;
    len += d;
    actions += w.dwell ? Math.round(GRIP_DWELL_S / 0.1) : Math.max(1, Math.ceil(d / MAX_STEP_MM));
  }
  return { len, actions };
}

export const SCHEMA = {
  frame: "cell: origin at the robot base on the table top, +X forward, +Y left, +Z up",
  units: { length: "mm", angle: "deg" },
  table: { x: TABLE_X, y: TABLE_Y },
  kinds: Object.fromEntries(Object.entries(KINDS).map(([id, k]) => [id, { label: k.label, height_mm: k.h, footprint_radius_mm: k.r, grip_width_mm: k.grip }])),
  bins: BINS.map((b) => ({ ...b, inner_floor_z: BIN_FLOOR, outer_mm: [50, 50, BIN_H] })),
  robot: { model: "FANUC ER-4iA", home_tcp: HOME, safe_z: SAFE_Z, speed_mm_s: SPEED_MM_S, max_step_mm: MAX_STEP_MM, gripper_open_mm: GRIP_OPEN },
  move_status: ["pending", "approved", "running", "done", "failed", "rejected", "cancelled"],
  modes: { sim: "the server animates the arm and moves parts itself", live: "agents report real poses; user moves wait for an agent to execute them" },
};

