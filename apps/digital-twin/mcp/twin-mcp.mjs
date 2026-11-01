#!/usr/bin/env node
/*
 * MCP server (stdio) for the A-Team cell twin. Dependency-free: newline-delimited
 * JSON-RPC 2.0 on stdin/stdout, wrapping the HTTP API in ../twin-api.js.
 *
 *   TWIN_URL     base URL of the local site     (default http://127.0.0.1:8090)
 *   AGENT_TOKEN  bearer token for /api          (required for protected routes)
 *   AGENT_NAME   shown on the page as the agent (default "mcp-agent")
 *
 * Register with Claude Code:  claude mcp add ateam-twin -- node mcp/twin-mcp.mjs
 */
import readline from "node:readline";

const BASE = (process.env.TWIN_URL || "http://127.0.0.1:8090").replace(/\/+$/, "");
const NAME = process.env.AGENT_NAME || "mcp-agent";
const TOKEN = process.env.AGENT_TOKEN || "";

async function api(method, route, body) {
  const r = await fetch(`${BASE}/api/${route}`, {
    method,
    headers: { Authorization: `Bearer ${TOKEN}`, "X-Agent-Name": NAME, ...(body ? { "Content-Type": "application/json" } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(`${method} /api/${route} → ${r.status}: ${data.error || r.statusText}`);
  return data;
}

const brief = (m) => m && { ...m, waypoints: m.waypoints?.map((w) => ({ name: w.name, tcp: w.tcp.map((v) => +v.toFixed(1)), grip: w.grip })) };

const num = { type: "number" };
const partSchema = {
  type: "object",
  properties: {
    id: { type: "string", description: "Stable part id, e.g. cube_big_1. New ids create parts." },
    kind: { type: "string", enum: ["cube_big", "cube_small", "cube_yellow", "screw_m6", "screw_pan"], description: "Required for new parts." },
    x: { ...num, description: "mm, cell frame (+X forward from the robot base)" },
    y: { ...num, description: "mm, cell frame (+Y left)" },
    z: { ...num, description: "mm above the table top (bottom of the part); 2 inside a bin" },
    yaw: { ...num, description: "deg about +Z" },
    bin: { type: ["string", "null"], description: "binA, binB or null" },
    carried: { type: "boolean", description: "true while the part is in the gripper" },
    confidence: { ...num, description: "0..1 detection confidence" },
    color: { type: "string", description: "#rrggbb override for any-colour cubes" },
  },
  required: ["id"],
};

const TOOLS = [
  {
    name: "get_cell_state",
    description: "Snapshot of the cell twin: mode (sim/live), every part pose, the robot TCP/joints/gripper, and recent moves. Cell frame: origin at the robot base on the table top, +X forward, +Y left, +Z up, mm and degrees.",
    inputSchema: { type: "object", properties: {} },
    run: async () => {
      const s = await api("GET", "state");
      return { ...s, moves: s.moves.map(({ waypoints, ...m }) => m) };
    },
  },
  {
    name: "get_schema",
    description: "Frame, units, part kinds with sizes, bin positions, robot limits and move statuses.",
    inputSchema: { type: "object", properties: {} },
    run: () => api("GET", "schema"),
  },
  {
    name: "set_part_poses",
    description: "Report real part locations (e.g. from a camera). Upserts by id; replace=true also deletes parts not listed. In live mode, a part reported within 6 mm of an open move's target completes that move and clears its ghost.",
    inputSchema: { type: "object", properties: { parts: { type: "array", items: partSchema, minItems: 1 }, replace: { type: "boolean" } }, required: ["parts"] },
    run: ({ parts, replace }) => api("PUT", "parts", { parts, replace: !!replace }),
  },
  {
    name: "remove_part",
    description: "Remove a part from the twin (it left the cell). Cancels its open moves.",
    inputSchema: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
    run: ({ id }) => api("DELETE", `parts/${encodeURIComponent(id)}`),
  },
  {
    name: "set_robot_state",
    description: "Report the real robot state (live mode only). tcp is [x,y,z] or [x,y,z,w,p,r] in UF0 mm/deg; joints are J1..J6 in degrees; gripper is the finger gap in mm (34 = open).",
    inputSchema: {
      type: "object",
      properties: {
        tcp: { type: "array", items: num, minItems: 3, maxItems: 6 },
        joints: { type: "array", items: num, minItems: 6, maxItems: 6 },
        gripper: num,
        state: { type: "string", description: "free text shown on the page, e.g. 'executing #3'" },
      },
    },
    run: (a) => api("PUT", "robot", a),
  },
  {
    name: "next_user_move",
    description: "Wait for the next move a user approved on the page (drag-and-drop) and, in live mode, claim it so it shows as running. Returns null if none arrives within wait_s. Execute it on the robot, report poses as you go, then call update_move with status done or failed.",
    inputSchema: { type: "object", properties: { wait_s: { type: "number", minimum: 0, maximum: 60 }, claim: { type: "boolean" } } },
    run: async ({ wait_s = 25, claim = true }) => {
      const { moves, mode } = await api("GET", `moves?status=approved&wait=${Math.min(60, Math.max(0, wait_s))}`);
      const m = moves[0];
      if (!m) return { move: null, mode };
      if (claim && mode === "live") return { move: brief(await api("POST", `moves/${m.id}/claim`)), mode, claimed: true };
      return { move: brief(m), mode, claimed: false, note: mode === "sim" ? "sim mode: the server executes moves itself; switch to live to claim them" : undefined };
    },
  },
  {
    name: "update_move",
    description: "Update a move: status running|done|failed|rejected|cancelled|approved, and/or phase (e.g. 'grasp') and a note. 'done' places the part at the target.",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "integer" },
        status: { type: "string", enum: ["approved", "running", "done", "failed", "rejected", "cancelled"] },
        phase: { type: "string" },
        note: { type: "string" },
      },
      required: ["id"],
    },
    run: ({ id, ...b }) => api("PATCH", `moves/${id}`, b).then(brief),
  },
  {
    name: "request_move",
    description: "Ask the cell to move a part to (x, y) on the table or into a bin. Validated with the same rules as a user drag (reach, collisions, bin capacity). Shows a ghost on the page.",
    inputSchema: {
      type: "object",
      properties: { partId: { type: "string" }, x: num, y: num, bin: { type: "string", enum: ["binA", "binB"] }, approve: { type: "boolean" } },
      required: ["partId"],
    },
    run: (a) => api("POST", "moves", a).then(brief),
  },
  {
    name: "set_mode",
    description: "'sim': the server animates the arm. 'live': agents report the real robot and parts, and user moves wait for an agent.",
    inputSchema: { type: "object", properties: { mode: { type: "string", enum: ["sim", "live"] } }, required: ["mode"] },
    run: ({ mode }) => api("PUT", "mode", { mode }),
  },
  {
    name: "reset_cell",
    description: "Reset parts to the default layout, clear all moves and home the robot.",
    inputSchema: { type: "object", properties: {} },
    run: async () => { await api("POST", "reset"); return { ok: true }; },
  },
];

// ───────── JSON-RPC over stdio ─────────
const send = (msg) => process.stdout.write(JSON.stringify({ jsonrpc: "2.0", ...msg }) + "\n");
const rl = readline.createInterface({ input: process.stdin });
rl.on("line", async (line) => {
  if (!line.trim()) return;
  let req;
  try { req = JSON.parse(line); } catch { return send({ id: null, error: { code: -32700, message: "parse error" } }); }
  const { id, method, params = {} } = req;
  if (id === undefined) return; // notification
  try {
    if (method === "initialize") {
      return send({ id, result: {
        protocolVersion: params.protocolVersion || "2025-06-18",
        capabilities: { tools: {} },
        serverInfo: { name: "ateam-twin", version: "1.0.0" },
        instructions: "Live twin of the A-Team FANUC ER-4iA cell. Report real part/robot poses with set_part_poses and set_robot_state (set_mode live first); pick up user drag-and-drop moves with next_user_move and finish them with update_move.",
      } });
    }
    if (method === "ping") return send({ id, result: {} });
    if (method === "tools/list") return send({ id, result: { tools: TOOLS.map(({ run, ...t }) => t) } });
    if (method === "tools/call") {
      const tool = TOOLS.find((t) => t.name === params.name);
      if (!tool) return send({ id, error: { code: -32602, message: `unknown tool ${params.name}` } });
      try {
        const out = await tool.run(params.arguments || {});
        return send({ id, result: { content: [{ type: "text", text: JSON.stringify(out, null, 1) }] } });
      } catch (err) {
        return send({ id, result: { content: [{ type: "text", text: String(err.message || err) }], isError: true } });
      }
    }
    send({ id, error: { code: -32601, message: `method not found: ${method}` } });
  } catch (err) {
    send({ id, error: { code: -32603, message: String(err.message || err) } });
  }
});
