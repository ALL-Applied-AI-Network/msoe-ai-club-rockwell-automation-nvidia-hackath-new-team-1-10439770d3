/*
 * Live state + HTTP API for the Drag-to-Move twin.
 *
 * The server owns the cell state. Browsers render it and post user moves;
 * agents (directly over HTTP or through mcp/twin-mcp.mjs) report real part and
 * robot poses and pick up user moves to execute. Changes fan out over SSE.
 *
 *   mode "sim":  the server animates the arm itself at the cell's 20 mm/s cap.
 *   mode "live": agent reports are the truth; a user move stays a ghost until an
 *                agent marks it done or reports the part at the target.
 */
"use strict";

const ID_RE = /^[A-Za-z0-9_.-]{1,48}$/;
const MAX_BODY = 256 * 1024;
const OPEN = new Set(["pending", "approved", "running"]);
const FINAL = new Set(["done", "failed", "rejected", "cancelled"]);

function createTwin(core) {
  const { KINDS, BINS, HOME, GRIP_OPEN, SPEED_MM_S, GRIP_DWELL_S, LIVE_DONE_TOL_MM } = core;

  const state = {
    rev: 0,
    mode: "sim",
    sim: { speed: 4, autoApprove: true },
    parts: {},
    moves: [],
    robot: { tcp: [...HOME, -180, 0, 0], gripper: GRIP_OPEN, joints: core.jointReadout(HOME), state: "idle", source: "sim", updatedAt: Date.now() },
    agent: { name: null, lastSeen: null, updates: 0 },
  };
  let moveSeq = 0;

  function resetParts() {
    state.parts = {};
    for (const p of core.defaultParts()) state.parts[p.id] = { ...p, carried: false, source: "layout", confidence: null, updatedAt: Date.now() };
  }
  resetParts();

  // ───────── SSE fan-out ─────────
  const clients = new Set();
  function emit(event, data) {
    state.rev++;
    const msg = `event: ${event}\ndata: ${JSON.stringify({ rev: state.rev, ...data })}\n\n`;
    for (const res of clients) res.write(msg);
  }
  const snapshot = () => ({
    mode: state.mode,
    sim: state.sim,
    parts: Object.values(state.parts),
    moves: state.moves.slice(-60).map(publicMove),
    robot: state.robot,
    agent: state.agent,
    clients: clients.size,
  });
  setInterval(() => { for (const res of clients) res.write(": keep-alive\n\n"); }, 15000).unref();

  const publicMove = (m) => ({ ...m });
  const emitPart = (p) => emit("part", { part: p });
  const emitMove = (m) => { emit("move", { move: publicMove(m) }); wakeWaiters(); };

  // ───────── Long-poll waiters for agents ─────────
  let waiters = [];
  function wakeWaiters() {
    waiters = waiters.filter((w) => {
      const hit = w.match();
      if (hit.length) { w.done(hit); return false; }
      return true;
    });
  }

  // ───────── Moves ─────────
  function createMove(partId, { x, y, bin, approve, origin }) {
    const p = state.parts[partId];
    if (!p) return { error: 404, why: `unknown part "${partId}"` };
    if (p.carried || state.moves.some((m) => m.partId === partId && OPEN.has(m.status))) return { error: 409, why: `${partId} already has an open move` };
    const to = core.resolveDrop(state, p, Number(x), Number(y), bin || null);
    if (!to.ok) return { error: 422, why: to.why, to };
    const m = {
      id: ++moveSeq,
      partId,
      kind: p.kind,
      label: KINDS[p.kind].label,
      from: { x: p.x, y: p.y, z: p.z, bin: p.bin },
      to: { x: to.x, y: to.y, z: to.z, bin: to.bin, where: to.where },
      status: approve ?? state.sim.autoApprove ? "approved" : "pending",
      phase: null,
      origin,
      note: null,
      waypoints: core.planMove(p, to),
      createdAt: Date.now(),
      updatedAt: Date.now(),
      elapsedS: 0,
    };
    m.stats = core.pathStats(state.robot.tcp, m.waypoints);
    state.moves.push(m);
    if (state.moves.length > 200) state.moves = state.moves.filter((x) => OPEN.has(x.status)).concat(state.moves.filter((x) => !OPEN.has(x.status)).slice(-100));
    emitMove(m);
    return { move: m };
  }

  function setMoveStatus(m, status, note) {
    if (FINAL.has(m.status)) return { error: 409, why: `move #${m.id} is already ${m.status}` };
    m.status = status;
    if (note !== undefined) m.note = String(note).slice(0, 300);
    m.updatedAt = Date.now();
    const p = state.parts[m.partId];
    if (status === "done" && p) {
      // Trust the ack: the part is at the target unless an agent reports otherwise.
      Object.assign(p, { x: m.to.x, y: m.to.y, z: m.to.z, bin: m.to.bin, carried: false, updatedAt: Date.now() });
      emitPart(p);
    }
    if ((status === "failed" || status === "cancelled" || status === "rejected") && p && p.carried) {
      p.carried = false;
      emitPart(p);
    }
    if (sim.move === m && status !== "running") sim.move = null;
    emitMove(m);
    return { move: m };
  }

  // ───────── Parts ─────────
  const num = (v, lim = 5000) => Number.isFinite(v) && Math.abs(v) <= lim;
  function upsertPart(input, source) {
    if (!input || !ID_RE.test(String(input.id || ""))) return { error: 400, why: "part.id must match [A-Za-z0-9_.-]{1,48}" };
    const existing = state.parts[input.id];
    const kind = input.kind ?? existing?.kind;
    if (!KINDS[kind]) return { error: 400, why: `part.kind must be one of ${Object.keys(KINDS).join(", ")}` };
    for (const f of ["x", "y", "z", "yaw"]) if (input[f] !== undefined && !num(input[f])) return { error: 400, why: `part.${f} must be a finite number` };
    if (!existing && (input.x === undefined || input.y === undefined)) return { error: 400, why: "new parts need x and y" };
    if (input.bin != null && !BINS.some((b) => b.id === input.bin)) return { error: 400, why: `part.bin must be null or one of ${BINS.map((b) => b.id).join(", ")}` };
    if (existing && state.mode === "sim" && source === "agent" && (existing.carried || state.moves.some((m) => m.partId === existing.id && m.status === "running")))
      return { error: 409, why: `${existing.id} is being moved by the simulator; switch to live mode to drive it` };
    const p = existing || { id: input.id, kind, x: 0, y: 0, z: 0, yaw: 0, bin: null, carried: false };
    for (const f of ["x", "y", "z", "yaw"]) if (input[f] !== undefined) p[f] = input[f];
    if (input.bin !== undefined) p.bin = input.bin;
    if (input.carried !== undefined) p.carried = !!input.carried;
    if (input.color !== undefined) p.color = /^#[0-9a-f]{6}$/i.test(input.color) ? input.color : undefined;
    p.kind = kind;
    p.confidence = input.confidence === undefined ? p.confidence ?? null : Math.max(0, Math.min(1, Number(input.confidence) || 0));
    p.source = source;
    p.updatedAt = Date.now();
    state.parts[p.id] = p;
    // Live mode: the part reaching its target closes the ghost.
    for (const m of state.moves) {
      if (m.partId === p.id && (m.status === "approved" || m.status === "running") && !p.carried &&
          Math.hypot(p.x - m.to.x, p.y - m.to.y) <= LIVE_DONE_TOL_MM) {
        m.status = "done"; m.note = "part reported at target"; m.updatedAt = Date.now();
        emitMove(m);
      }
    }
    emitPart(p);
    return { part: p };
  }

  // ───────── Robot ─────────
  function setRobot(input, source) {
    const r = state.robot;
    if (input.tcp !== undefined) {
      if (!Array.isArray(input.tcp) || input.tcp.length < 3 || input.tcp.length > 6 || !input.tcp.every((v) => num(v))) return { error: 400, why: "robot.tcp must be [x, y, z] or [x, y, z, w, p, r] in mm/deg" };
      r.tcp = [...input.tcp, ...[-180, 0, 0].slice(input.tcp.length - 3)].slice(0, 6);
    }
    if (input.joints !== undefined) {
      if (!Array.isArray(input.joints) || input.joints.length !== 6 || !input.joints.every((v) => num(v, 720))) return { error: 400, why: "robot.joints must be six angles in degrees" };
      r.joints = input.joints;
    } else if (input.tcp !== undefined) r.joints = core.jointReadout(r.tcp);
    if (input.gripper !== undefined) {
      if (!num(input.gripper, 200)) return { error: 400, why: "robot.gripper is the finger gap in mm" };
      r.gripper = input.gripper;
    }
    if (input.state !== undefined) r.state = String(input.state).slice(0, 80);
    r.source = source;
    r.updatedAt = Date.now();
    emit("robot", { robot: r });
    return { robot: r };
  }

  // ───────── Simulator (mode "sim") ─────────
  const sim = { move: null, seg: 0, dwell: 0, last: Date.now(), lastEmit: 0 };
  function moveToward(target, dt) {
    const tcp = state.robot.tcp;
    const d = [0, 1, 2].map((i) => target[i] - tcp[i]);
    const dist = Math.hypot(...d);
    const step = SPEED_MM_S * dt;
    if (dist <= step) { tcp[0] = target[0]; tcp[1] = target[1]; tcp[2] = target[2]; return true; }
    for (let i = 0; i < 3; i++) tcp[i] += (d[i] / dist) * step;
    return false;
  }
  function simTick() {
    const now = Date.now();
    const dt = Math.min(0.25, (now - sim.last) / 1000) * state.sim.speed;
    sim.last = now;
    if (state.mode !== "sim") return;
    const r = state.robot;
    let changed = false;

    if (!sim.move) {
      const next = state.moves.find((m) => m.status === "approved");
      if (next) {
        Object.assign(sim, { move: next, seg: 0, dwell: 0 });
        const moving = state.parts[next.partId];
        next.cargo = moving && KINDS[moving.kind].isBin
          ? Object.values(state.parts).filter((q) => q.bin === moving.id).map((q) => ({ id: q.id, dx: q.x - moving.x, dy: q.y - moving.y, dz: q.z - moving.z }))
          : [];
        next.status = "running"; next.elapsedS = 0; next.updatedAt = now;
        next.stats = core.pathStats(r.tcp, next.waypoints);
        emitMove(next);
      } else {
        const away = Math.hypot(HOME[0] - r.tcp[0], HOME[1] - r.tcp[1], HOME[2] - r.tcp[2]);
        if (away > 0.5) { moveToward(HOME, dt); changed = true; }
        const st = away > 0.5 ? "returning home" : "idle";
        if (r.state !== st) { r.state = st; changed = true; }
      }
    }
    const m = sim.move;
    if (m) {
      const w = m.waypoints[sim.seg];
      const p = state.parts[m.partId];
      m.elapsedS += dt;
      r.state = `#${m.id} ${w.name}`;
      if (m.phase !== w.name) { m.phase = w.name; emitMove(m); }
      if (w.dwell) {
        sim.dwell += dt;
        const target = w.grip ? KINDS[m.kind].grip : GRIP_OPEN;
        r.gripper += (target - r.gripper) * Math.min(1, dt * 8);
        if (sim.dwell >= GRIP_DWELL_S) {
          sim.dwell = 0;
          r.gripper = target;
          if (p) {
            if (w.grip) p.carried = true;
            else Object.assign(p, { carried: false, x: m.to.x, y: m.to.y, z: m.to.z, bin: m.to.bin });
            p.source = "sim"; p.updatedAt = now;
            emitPart(p);
          }
          sim.seg++;
        }
      } else if (moveToward(w.tcp, dt)) sim.seg++;
      if (p && p.carried) {
        p.x = r.tcp[0]; p.y = r.tcp[1]; p.z = r.tcp[2] - KINDS[p.kind].h / 2; p.bin = null;
        if (now - sim.lastEmit > 45) emitPart(p);
        for (const c of m.cargo || []) {
          const q = state.parts[c.id];
          if (!q) continue;
          q.x = p.x + c.dx; q.y = p.y + c.dy; q.z = p.z + c.dz; q.bin = p.id;
          q.source = "sim"; q.updatedAt = now;
          if (now - sim.lastEmit > 45) emitPart(q);
        }
      }
      if (sim.seg >= m.waypoints.length) {
        m.status = "done"; m.phase = null; m.updatedAt = now;
        sim.move = null;
        emitMove(m);
      }
      changed = true;
    }
    if (changed && now - sim.lastEmit > 45) {
      sim.lastEmit = now;
      r.joints = core.jointReadout(r.tcp);
      r.source = "sim";
      r.updatedAt = now;
      emit("robot", { robot: r });
    }
  }
  setInterval(simTick, 33).unref();

  function setMode(mode) {
    if (mode !== "sim" && mode !== "live") return { error: 400, why: 'mode must be "sim" or "live"' };
    if (mode === state.mode) return { mode };
    state.mode = mode;
    // Hand any in-flight move back to the queue so the new executor can take it.
    for (const m of state.moves) if (m.status === "running") { m.status = "approved"; m.phase = null; m.updatedAt = Date.now(); }
    for (const p of Object.values(state.parts)) p.carried = false;
    sim.move = null;
    state.robot.source = mode === "sim" ? "sim" : state.robot.source;
    emit("snapshot", snapshot());
    wakeWaiters();
    return { mode };
  }

  // ───────── HTTP ─────────
  function send(res, status, body) {
    res.writeHead(status, { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" });
    res.end(JSON.stringify(body));
  }
  function readJson(req) {
    return new Promise((resolve, reject) => {
      if (!/^application\/json\b/i.test(req.headers["content-type"] || "")) return reject(Object.assign(new Error("Content-Type must be application/json"), { status: 415 }));
      let body = "";
      req.on("data", (c) => { body += c; if (body.length > MAX_BODY) { reject(Object.assign(new Error("body too large"), { status: 413 })); req.destroy(); } });
      req.on("end", () => { try { resolve(body ? JSON.parse(body) : {}); } catch { reject(Object.assign(new Error("invalid JSON"), { status: 400 })); } });
      req.on("error", reject);
    });
  }
  const fail = (res, r) => send(res, r.error, { error: r.why, ...(r.to ? { target: r.to } : {}) });

  function noteAgent(actor, req) {
    if (actor !== "agent") return;
    state.agent.name = String(req.headers["x-agent-name"] || "agent").slice(0, 60);
    state.agent.lastSeen = Date.now();
    state.agent.updates++;
    emit("agent", { agent: state.agent });
  }

  async function handle(req, res, url, actor) {
    const parts = url.pathname.split("/").filter(Boolean); // ["api", ...]
    const [, res1, id, sub] = parts;
    const M = req.method;
    try {
      if (res1 === "schema" && M === "GET") return send(res, 200, core.SCHEMA);
      if (res1 === "state" && M === "GET") return send(res, 200, snapshot());

      if (res1 === "events" && M === "GET") {
        res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-store", Connection: "keep-alive", "X-Accel-Buffering": "no" });
        res.write("retry: 2000\n\n");
        clients.add(res);
        res.write(`event: snapshot\ndata: ${JSON.stringify({ rev: state.rev, ...snapshot() })}\n\n`);
        req.on("close", () => clients.delete(res));
        return;
      }

      if (res1 === "parts") {
        if (!id && M === "GET") return send(res, 200, { parts: Object.values(state.parts) });
        if (!id && (M === "PUT" || M === "POST")) {
          const body = await readJson(req);
          const list = Array.isArray(body) ? body : body.parts;
          if (!Array.isArray(list) || list.length > 500) return send(res, 400, { error: "send { parts: [...] } (max 500)" });
          const out = [], errors = [];
          if (body.replace === true) {
            const keep = new Set(list.map((p) => p.id));
            for (const pid of Object.keys(state.parts)) if (!keep.has(pid)) { delete state.parts[pid]; emit("part_removed", { id: pid }); }
          }
          for (const item of list) {
            const r = upsertPart(item, actor);
            r.error ? errors.push({ id: item?.id, error: r.why }) : out.push(r.part);
          }
          noteAgent(actor, req);
          return send(res, errors.length && !out.length ? 400 : 200, { updated: out.length, parts: out, errors });
        }
        if (id && M === "GET") return state.parts[id] ? send(res, 200, state.parts[id]) : send(res, 404, { error: "unknown part" });
        if (id && M === "PATCH") {
          if (!state.parts[id]) return send(res, 404, { error: "unknown part" });
          const r = upsertPart({ ...(await readJson(req)), id }, actor);
          noteAgent(actor, req);
          return r.error ? fail(res, r) : send(res, 200, r.part);
        }
        if (id && M === "DELETE") {
          if (!state.parts[id]) return send(res, 404, { error: "unknown part" });
          delete state.parts[id];
          for (const m of state.moves) if (m.partId === id && OPEN.has(m.status)) setMoveStatus(m, "cancelled", "part removed");
          emit("part_removed", { id });
          return send(res, 200, { removed: id });
        }
      }

      if (res1 === "robot") {
        if (M === "GET") return send(res, 200, state.robot);
        if (M === "PUT" || M === "POST" || M === "PATCH") {
          if (state.mode === "sim") return send(res, 409, { error: 'the simulator drives the robot in "sim" mode; PUT /api/mode {"mode":"live"} first' });
          const r = setRobot(await readJson(req), actor);
          noteAgent(actor, req);
          return r.error ? fail(res, r) : send(res, 200, r.robot);
        }
      }

      if (res1 === "moves") {
        if (!id && M === "GET") {
          const status = url.searchParams.get("status");
          const since = Number(url.searchParams.get("since") || 0);
          const wait = Math.min(60, Math.max(0, Number(url.searchParams.get("wait") || 0)));
          const match = () => state.moves.filter((m) => m.id > since && (!status || status.split(",").includes(m.status))).map(publicMove);
          const now = match();
          if (now.length || !wait) return send(res, 200, { moves: now, mode: state.mode });
          const w = { match, done: (list) => { clearTimeout(w.t); send(res, 200, { moves: list, mode: state.mode }); } };
          w.t = setTimeout(() => { waiters = waiters.filter((x) => x !== w); send(res, 200, { moves: [], mode: state.mode }); }, wait * 1000);
          req.on("close", () => { clearTimeout(w.t); waiters = waiters.filter((x) => x !== w); });
          waiters.push(w);
          return;
        }
        if (!id && M === "POST") {
          const b = await readJson(req);
          const r = createMove(String(b.partId || b.part || ""), { x: b.x, y: b.y, bin: b.bin, approve: b.approve, origin: actor });
          noteAgent(actor, req);
          return r.error ? fail(res, r) : send(res, 201, r.move);
        }
        const m = state.moves.find((x) => x.id === Number(id));
        if (id && !m) return send(res, 404, { error: "unknown move" });
        if (m && M === "GET") return send(res, 200, m);
        if (m && sub === "claim" && M === "POST") {
          if (state.mode !== "live") return send(res, 409, { error: 'moves are executed by the simulator in "sim" mode' });
          if (m.status !== "approved") return send(res, 409, { error: `move #${m.id} is ${m.status}, not approved` });
          noteAgent(actor, req);
          const r = setMoveStatus(m, "running");
          return r.error ? fail(res, r) : send(res, 200, r.move);
        }
        if (m && M === "PATCH") {
          const b = await readJson(req);
          const allowed = actor === "agent" ? ["approved", "running", "done", "failed", "rejected", "cancelled"] : ["approved", "rejected", "cancelled"];
          if (b.phase !== undefined && b.status === undefined) {
            m.phase = b.phase == null ? null : String(b.phase).slice(0, 40);
            m.updatedAt = Date.now();
            emitMove(m);
            noteAgent(actor, req);
            return send(res, 200, m);
          }
          if (!allowed.includes(b.status)) return send(res, 400, { error: `status must be one of ${allowed.join(", ")}` });
          if (b.status === "approved" && m.status !== "pending") return send(res, 409, { error: `move #${m.id} is ${m.status}` });
          if (b.phase !== undefined) m.phase = b.phase;
          noteAgent(actor, req);
          const r = setMoveStatus(m, b.status, b.note);
          return r.error ? fail(res, r) : send(res, 200, r.move);
        }
      }

      if (res1 === "mode") {
        if (M === "GET") return send(res, 200, { mode: state.mode });
        if (M === "PUT" || M === "POST") {
          const r = setMode((await readJson(req)).mode);
          noteAgent(actor, req);
          return r.error ? fail(res, r) : send(res, 200, r);
        }
      }

      if (res1 === "sim" && (M === "PUT" || M === "POST")) {
        const b = await readJson(req);
        if (b.speed !== undefined) {
          if (![1, 2, 4, 10, 20].includes(b.speed)) return send(res, 400, { error: "speed must be 1, 2, 4, 10 or 20" });
          state.sim.speed = b.speed;
        }
        if (b.autoApprove !== undefined) {
          state.sim.autoApprove = !!b.autoApprove;
          if (state.sim.autoApprove) for (const m of state.moves) if (m.status === "pending") setMoveStatus(m, "approved");
        }
        emit("sim", { sim: state.sim });
        return send(res, 200, state.sim);
      }

      if (res1 === "reset" && M === "POST") {
        for (const m of state.moves) if (OPEN.has(m.status)) { m.status = "cancelled"; m.note = "scene reset"; }
        state.moves = [];
        moveSeq = 0;
        sim.move = null;
        resetParts();
        Object.assign(state.robot, { tcp: [...HOME, -180, 0, 0], gripper: GRIP_OPEN, joints: core.jointReadout(HOME), state: "idle", source: state.mode === "sim" ? "sim" : actor, updatedAt: Date.now() });
        emit("snapshot", snapshot());
        wakeWaiters();
        return send(res, 200, snapshot());
      }

      return send(res, 404, { error: `no route ${M} ${url.pathname}` });
    } catch (err) {
      return send(res, err.status || 500, { error: err.status ? err.message : "internal error" });
    }
  }

  return { handle, state };
}

module.exports = { createTwin };
