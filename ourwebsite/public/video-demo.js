/*
 * Demo 1 — "Show the Job": recorded video → FANUC TCP plan.
 *
 * Real (runs in the browser): frame sampling, motion-energy segmentation,
 * keyframe extraction, color-blob grounding, waypoint interpolation limits.
 * Simulated: the step parser that a vision-language model will replace, and
 * the image→robot-frame mapping (no camera calibration is applied).
 */
(() => {
  const SAMPLE_SRC = "assets/testbed-demo.mp4";
  const SAMPLE_CAPTION = "Sample: official testbed side camera, Oct 19 trial (operator-ended). Upload any phone video of a pick-and-place to try your own.";
  const N_SAMPLES = 48;
  const N_KEYFRAMES = 8;

  // Cell interface, matching robot-mvp/policy.py and the Oct 20 commissioning run.
  const DT_S = 0.1;
  const MAX_STEP_MM = 2.0;
  const HOME = [300.0, 0.0, -90.0, -180.0, 0.0, 0.0, 0.0];
  const SAFE_Z = -230.0;     // approved transit height from the commissioning plan
  const GRASP_Z = -285.0;    // demo value — set from calibrated table height
  const PLACE_Z = -280.0;
  const ENVELOPE = { x: [250, 600], y: [-250, 250], z: [-300, 0] }; // demo envelope

  const $ = (id) => document.getElementById(id);
  const player = $("vPlayer");
  const overlay = $("vOverlay");
  const energyCanvas = $("energy");
  const filmstrip = $("filmstrip");
  const runBtn = $("vRun");
  const stages = [...document.querySelectorAll("#vStages li")];

  let source = { src: SAMPLE_SRC, name: "testbed-demo.mp4", sample: true };
  let result = null;
  let running = false;

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const fmtT = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
  const f1 = (v) => v.toFixed(1);

  // ───────── Source selection ─────────
  $("vSample").addEventListener("click", () => {
    if (running) return;
    setSource({ src: SAMPLE_SRC, name: "testbed-demo.mp4", sample: true });
    $("vCaption").textContent = SAMPLE_CAPTION;
  });

  $("vFile").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file || running) return;
    if (source.url) URL.revokeObjectURL(source.url);
    const url = URL.createObjectURL(file);
    setSource({ src: url, url, name: file.name, sample: false });
    $("vCaption").textContent = `Your upload: ${file.name} (${(file.size / 1e6).toFixed(1)} MB). Processed locally; nothing leaves your browser.`;
    e.target.value = "";
  });

  function setSource(next) {
    source = next;
    player.removeAttribute("poster");
    if (next.sample) player.poster = "assets/testbed-poster.jpg";
    player.src = next.src;
    $("vSample").classList.toggle("is-active", next.sample);
    document.querySelector('label[for="vFile"]').classList.toggle("is-active", !next.sample);
    reset();
  }

  function reset() {
    result = null;
    stages.forEach((li) => li.classList.remove("is-running", "is-done", "is-wait"));
    filmstrip.innerHTML = "";
    $("kfInfo").textContent = "—";
    $("energyInfo").textContent = "computed in your browser";
    $("vClock").textContent = "idle";
    $("vGate").hidden = true;
    $("outSteps").innerHTML = '<li class="empty">Run the conversion to generate step-by-step instructions.</li>';
    $("outWaypoints").innerHTML = '<tr><td colspan="9" class="empty">No waypoints yet.</td></tr>';
    $("outPlan").textContent = "// plan appears here";
    $("outTP").textContent = "! TP program appears here";
    clearCanvas(overlay);
    drawEnergy(null);
    drawTopView(null);
  }

  // ───────── Tabs ─────────
  document.querySelectorAll("#show .tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll("#show .tab").forEach((t) => {
        t.classList.toggle("is-active", t === tab);
        t.setAttribute("aria-selected", t === tab);
      });
      document.querySelectorAll("#show .tab-pane").forEach((p) => p.classList.toggle("is-active", p.dataset.pane === tab.dataset.tab));
      if (tab.dataset.tab === "waypoints") drawTopView(result);
    });
  });

  // ───────── Pipeline ─────────
  runBtn.addEventListener("click", run);

  async function run() {
    if (running) return;
    running = true;
    runBtn.disabled = true;
    runBtn.textContent = "Converting…";
    const t0 = performance.now();
    const clock = setInterval(() => ($("vClock").textContent = `${((performance.now() - t0) / 1000).toFixed(1)} s`), 100);
    reset();
    $("vClock").textContent = "0.0 s";

    try {
      setStage("sample", "running");
      const frames = await sampleFrames(source.src);
      setStage("sample", "done");

      setStage("segment", "running");
      const seg = segment(frames);
      drawEnergy(seg);
      renderKeyframes(frames, seg);
      await sleep(450);
      setStage("segment", "done");

      setStage("ground", "running");
      const ground = groundObjects(frames);
      drawOverlay(ground, frames.width, frames.height);
      await sleep(550);
      setStage("ground", "done");

      setStage("plan", "running");
      const plan = buildPlan(ground);
      await sleep(500);
      setStage("plan", "done");

      setStage("safety", "running");
      plan.safety = checkSafety(plan.waypoints);
      await sleep(400);
      setStage("safety", "done");

      result = { seg, ground, plan, frames };
      renderOutputs(result);
      setStage("gate", "wait");
      $("vGate").hidden = false;
    } catch (err) {
      console.error(err);
      stages.forEach((li) => li.classList.remove("is-running"));
      $("outSteps").innerHTML = `<li class="empty">Couldn't decode this video in your browser (${err.message}). Try an MP4 (H.264) or WebM file.</li>`;
    } finally {
      clearInterval(clock);
      $("vClock").textContent = `${((performance.now() - t0) / 1000).toFixed(1)} s`;
      running = false;
      runBtn.disabled = false;
      runBtn.textContent = "↻ Convert again";
    }
  }

  function setStage(name, state) {
    const li = stages.find((l) => l.dataset.stage === name);
    li.classList.remove("is-running", "is-done", "is-wait");
    li.classList.add(`is-${state}`);
  }

  // ───────── Frame sampling (real) ─────────
  function sampleFrames(src) {
    return new Promise((resolve, reject) => {
      const v = document.createElement("video");
      v.muted = true;
      v.playsInline = true;
      v.preload = "auto";
      v.src = src;
      const W = 160, H = 90;
      const small = document.createElement("canvas");
      small.width = W; small.height = H;
      const sctx = small.getContext("2d", { willReadFrequently: true });
      const out = { width: W, height: H, times: [], pixels: [], thumbs: [], duration: 0, aspect: 16 / 9 };

      v.addEventListener("error", () => reject(new Error("unsupported codec or file")), { once: true });
      v.addEventListener("loadedmetadata", async () => {
        try {
          const dur = Number.isFinite(v.duration) ? v.duration : 0;
          if (!dur) throw new Error("video has no duration");
          out.duration = dur;
          out.aspect = v.videoWidth / v.videoHeight || 16 / 9;
          for (let i = 0; i < N_SAMPLES; i++) {
            const t = Math.min(dur - 0.05, (i + 0.5) * (dur / N_SAMPLES));
            await seek(v, t);
            sctx.drawImage(v, 0, 0, W, H);
            out.times.push(t);
            out.pixels.push(sctx.getImageData(0, 0, W, H).data);
            const thumb = document.createElement("canvas");
            thumb.width = 192; thumb.height = 108;
            thumb.getContext("2d").drawImage(v, 0, 0, 192, 108);
            out.thumbs.push(thumb);
            $("kfInfo").textContent = `${i + 1}/${N_SAMPLES}`;
          }
          resolve(out);
        } catch (e) { reject(e); }
      }, { once: true });
    });
  }

  function seek(v, t) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("seek timed out")), 8000);
      v.addEventListener("seeked", () => { clearTimeout(timer); resolve(); }, { once: true });
      v.currentTime = t;
    });
  }

  // ───────── Motion segmentation (real) ─────────
  function segment(frames) {
    const n = frames.pixels.length;
    const raw = new Array(n).fill(0);
    for (let i = 1; i < n; i++) {
      const a = frames.pixels[i - 1], b = frames.pixels[i];
      let sum = 0;
      for (let p = 0; p < a.length; p += 4) {
        const ga = a[p] * 0.299 + a[p + 1] * 0.587 + a[p + 2] * 0.114;
        const gb = b[p] * 0.299 + b[p + 1] * 0.587 + b[p + 2] * 0.114;
        sum += Math.abs(ga - gb);
      }
      raw[i] = sum / (a.length / 4);
    }
    raw[0] = raw[1] || 0;
    const energy = raw.map((_, i) => {
      const w = raw.slice(Math.max(0, i - 1), i + 2);
      return w.reduce((s, x) => s + x, 0) / w.length;
    });
    const lo = Math.min(...energy), hi = Math.max(...energy);
    const thr = lo + 0.3 * (hi - lo);

    // Contiguous active runs, merging gaps shorter than two samples.
    const runs = [];
    let start = -1;
    energy.forEach((e, i) => {
      if (e >= thr && start < 0) start = i;
      if ((e < thr || i === n - 1) && start >= 0) {
        const end = e >= thr ? i : i - 1;
        const prev = runs[runs.length - 1];
        if (prev && start - prev[1] <= 2) prev[1] = end;
        else runs.push([start, end]);
        start = -1;
      }
    });

    // Step times: equal shares of cumulative motion, so steps cluster where the demonstrator moves.
    const total = energy.reduce((s, x) => s + x, 0) || 1;
    const cdf = [];
    energy.reduce((acc, e, i) => (cdf[i] = acc + e / total), 0);
    const stepTimes = STEP_TEMPLATE.map((_, k) => {
      const target = (k + 0.5) / STEP_TEMPLATE.length;
      const idx = Math.max(0, cdf.findIndex((c) => c >= target));
      return frames.times[idx];
    });
    const stepConf = STEP_TEMPLATE.map((_, k) => {
      const idx = frames.times.indexOf(stepTimes[k]);
      return 0.72 + 0.25 * ((energy[idx] - lo) / ((hi - lo) || 1));
    });

    // Keyframes: highest-energy sample inside each of N equal windows.
    const keyIdx = [];
    for (let k = 0; k < N_KEYFRAMES; k++) {
      const a = Math.floor((k * n) / N_KEYFRAMES), b = Math.floor(((k + 1) * n) / N_KEYFRAMES);
      let best = a;
      for (let i = a; i < b; i++) if (energy[i] > energy[best]) best = i;
      keyIdx.push(best);
    }
    return { energy, thr, runs, stepTimes, stepConf, keyIdx, lo, hi, duration: frames.duration, times: frames.times };
  }

  // ───────── Color-blob grounding (real detection, uncalibrated mapping) ─────────
  function groundObjects(frames) {
    const W = frames.width, H = frames.height;
    const px = frames.pixels[0];
    const find = (test) => {
      let sx = 0, sy = 0, c = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const p = (y * W + x) * 4;
        if (test(px[p], px[p + 1], px[p + 2])) { sx += x; sy += y; c++; }
      }
      return c >= 3 ? { u: sx / c / W, v: sy / c / H, n: c } : null;
    };
    const red = find((r, g, b) => r > 120 && r > g * 1.7 && r > b * 1.7);
    const blue = find((r, g, b) => b > 100 && b > r * 1.5 && b > g * 1.2);
    const toUF0 = (o) => ({
      x: clamp(560 - o.v * 220, 380, 540),
      y: clamp((o.u - 0.5) * 340, -150, 150),
    });
    const pick = red ? { ...toUF0(red), label: "red part", img: red, found: true }
                     : { x: 450, y: -70, label: "part", img: null, found: false };
    let place = blue ? { ...toUF0(blue), label: "blue marker", img: blue, found: true }
                     : { x: 450, y: 70, label: "target B", img: null, found: false };
    if (blue) place.y = clamp(place.y + 45, -150, 150); // place beside the marker, not on it
    if (Math.hypot(place.x - pick.x, place.y - pick.y) < 60) place = { ...place, y: clamp(pick.y + 140, -150, 150) };
    return { pick, place };
  }

  // ───────── Planning ─────────
  const STEP_TEMPLATE = [
    { id: "approach", title: "Approach the part", detail: (g) => `Move above the ${g.pick.label} at safe height Z ${SAFE_Z} mm, gripper open.` },
    { id: "descend", title: "Descend to grasp", detail: () => `Straight-line descent to Z ${GRASP_Z} mm; tool pointing down (W −180°).` },
    { id: "grasp", title: "Close gripper", detail: () => "Close the SMC MHZL2-16D2 fingers; confirm grip before lifting." },
    { id: "lift", title: "Lift clear", detail: () => `Retract vertically to Z ${SAFE_Z} mm before any lateral move.` },
    { id: "transfer", title: "Transfer to target", detail: (g) => `Carry at safe height to the ${g.place.label === "blue marker" ? "spot beside the blue marker" : g.place.label}.` },
    { id: "place", title: "Lower and place", detail: () => `Descend to Z ${PLACE_Z} mm over the target.` },
    { id: "release", title: "Release", detail: () => "Open the gripper and pause for the part to settle." },
    { id: "retreat", title: "Retreat home", detail: () => "Rise to safe height, then return to the home pose." },
  ];

  function buildPlan(g) {
    const P = g.pick, T = g.place;
    const o = [-180.0, 0.0, 0.0];
    const wp = [
      { step: "home", pose: [...HOME] },
      { step: "approach", pose: [P.x, P.y, SAFE_Z, ...o, 0] },
      { step: "descend", pose: [P.x, P.y, GRASP_Z, ...o, 0] },
      { step: "grasp", pose: [P.x, P.y, GRASP_Z, ...o, 1] },
      { step: "lift", pose: [P.x, P.y, SAFE_Z, ...o, 1] },
      { step: "transfer", pose: [T.x, T.y, SAFE_Z, ...o, 1] },
      { step: "place", pose: [T.x, T.y, PLACE_Z, ...o, 1] },
      { step: "release", pose: [T.x, T.y, PLACE_Z, ...o, 0] },
      { step: "retreat", pose: [T.x, T.y, SAFE_Z, ...o, 0] },
      { step: "home", pose: [...HOME] },
    ].map((w) => ({ ...w, pose: w.pose.map((v) => Math.round(v * 10) / 10) }));

    let length = 0, actions = 0;
    for (let i = 1; i < wp.length; i++) {
      const d = Math.hypot(...[0, 1, 2].map((k) => wp[i].pose[k] - wp[i - 1].pose[k]));
      length += d;
      actions += Math.max(1, Math.ceil(d / MAX_STEP_MM));
    }
    return { waypoints: wp, length, actions, cycle: actions * DT_S };
  }

  function checkSafety(wp) {
    const issues = [];
    wp.forEach((w, i) => {
      const [x, y, z] = w.pose;
      if (x < ENVELOPE.x[0] || x > ENVELOPE.x[1]) issues.push(`P${i} X out of envelope`);
      if (y < ENVELOPE.y[0] || y > ENVELOPE.y[1]) issues.push(`P${i} Y out of envelope`);
      if (z < ENVELOPE.z[0] || z > ENVELOPE.z[1]) issues.push(`P${i} Z out of envelope`);
    });
    return { ok: issues.length === 0, issues };
  }

  // ───────── Rendering ─────────
  function renderOutputs({ seg, ground, plan }) {
    const steps = $("outSteps");
    steps.innerHTML = "";
    STEP_TEMPLATE.forEach((s, k) => {
      const li = document.createElement("li");
      li.dataset.t = seg.stepTimes[k];
      li.style.animationDelay = `${k * 60}ms`;
      li.style.cursor = "pointer";
      li.title = "Jump to this moment in the video";
      li.innerHTML = `<div><div class="s-title">${s.title}</div><div class="s-detail">${s.detail(ground)}</div>
        <span class="conf">confidence ${(seg.stepConf[k] * 100).toFixed(0)}%</span></div>
        <span class="s-time">@ ${fmtT(seg.stepTimes[k])}</span>`;
      li.addEventListener("click", () => { player.currentTime = seg.stepTimes[k]; player.play().catch(() => {}); });
      steps.appendChild(li);
    });
    const note = document.createElement("li");
    note.className = "empty";
    note.textContent = `${plan.waypoints.length} waypoints · ${plan.length.toFixed(0)} mm path · ${plan.actions} TCP actions · ≈${plan.cycle.toFixed(0)} s at 20 mm/s · envelope ${plan.safety.ok ? "✓ pass" : "✗ " + plan.safety.issues.join(", ")}` +
      (ground.pick.found ? "" : " · no colored part found, using default pick point");
    steps.appendChild(note);

    $("outWaypoints").innerHTML = plan.waypoints.map((w, i) => {
      const [x, y, z, ww, p, r, g] = w.pose;
      return `<tr><td>P${i}</td><td>${w.step}</td><td>${f1(x)}</td><td>${f1(y)}</td><td>${f1(z)}</td><td>${f1(ww)}</td><td>${f1(p)}</td><td>${f1(r)}</td>
        <td class="${g ? "grip-c" : "grip-o"}">${g ? "close" : "open"}</td></tr>`;
    }).join("");

    const planJson = {
      plan_id: `demo-${Date.now().toString(36)}`,
      status: "pending_human_approval",
      generated_by: "a-team/show-the-job (demo mode)",
      source_video: { name: source.name, duration_s: +seg.duration.toFixed(2), sampled_frames: seg.times.length },
      robot: "FANUC ER-4iA",
      policy_action_mode: "tcp",
      controller_frame: "UF0",
      action_format: ["x_mm", "y_mm", "z_mm", "w_deg", "p_deg", "r_deg", "gripper"],
      motion_limits: { sample_period_s: DT_S, maximum_translation_per_action_mm: MAX_STEP_MM, maximum_rotation_per_action_deg: 1.0 },
      grounding: {
        pick: { label: ground.pick.label, detected: ground.pick.found },
        place: { label: ground.place.label, detected: ground.place.found },
        calibration: "none — image→UF0 mapping is approximate in demo mode",
      },
      steps: STEP_TEMPLATE.map((s, k) => ({ id: s.id, instruction: s.title, video_t_s: +seg.stepTimes[k].toFixed(2), confidence: +seg.stepConf[k].toFixed(2) })),
      waypoints: plan.waypoints.map((w, i) => ({ name: `P${i}_${w.step}`, tcp: w.pose })),
      estimate: { path_mm: +plan.length.toFixed(1), actions: plan.actions, cycle_s: +plan.cycle.toFixed(1) },
      safety_check: plan.safety,
    };
    result.planJson = planJson;
    $("outPlan").innerHTML = highlightJson(JSON.stringify(planJson, null, 2));
    $("outTP").innerHTML = highlightTP(buildTP(plan));
    drawTopView(result);
  }

  function buildTP(plan) {
    const lines = [];
    let ln = 1;
    const L = (s) => lines.push(`${String(ln++).padStart(4)}:  ${s} ;`);
    L("!A-Team demo: generated from video") ;
    L("!DRAFT - requires operator approval");
    L("UFRAME_NUM=0");
    L("UTOOL_NUM=1");
    let grip = 0;
    plan.waypoints.forEach((w, i) => {
      const g = w.pose[6];
      if (g !== grip) {
        L(g ? "RO[1]=ON" : "RO[1]=OFF");
        L("WAIT   0.50(sec)");
        grip = g;
      } else {
        L(`${i === 0 ? "J" : "L"} P[${i + 1}:${w.step}] ${i === 0 ? "20%" : "20mm/sec"} FINE`);
      }
    });
    const pos = plan.waypoints.map((w, i) => {
      const [x, y, z, ww, p, r] = w.pose;
      return `P[${i + 1}:"${w.step}"]{\n   GP1:\n\tUF : 0, UT : 1,\tCONFIG : 'N U T, 0, 0, 0',\n\tX = ${f1(x).padStart(8)}  mm,\tY = ${f1(y).padStart(8)}  mm,\tZ = ${f1(z).padStart(8)}  mm,\n\tW = ${f1(ww).padStart(8)} deg,\tP = ${f1(p).padStart(8)} deg,\tR = ${f1(r).padStart(8)} deg\n};`;
    });
    return `/PROG  ATEAM_DEMO\n/ATTR\nCOMMENT\t\t= "SHOW THE JOB";\nDEFAULT_GROUP\t= 1,*,*,*,*;\n/MN\n${lines.join("\n")}\n/POS\n${pos.join("\n")}\n/END`;
  }

  function escapeHtml(s) { return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]); }
  function highlightJson(s) {
    return escapeHtml(s)
      .replace(/("(?:[^"\\]|\\.)*")(\s*:)/g, '<span class="k">$1</span>$2')
      .replace(/(:\s*|\[\s*|,\s*)("(?:[^"\\]|\\.)*")/g, '$1<span class="s">$2</span>')
      .replace(/(-?\b\d+(?:\.\d+)?\b)(?![^<]*<\/span>)/g, '<span class="n">$1</span>');
  }
  function highlightTP(s) {
    return escapeHtml(s)
      .replace(/(![^;\n]*)/g, '<span class="c">$1</span>')
      .replace(/^(\/\w+)/gm, '<span class="k">$1</span>')
      .replace(/\b(UFRAME_NUM|UTOOL_NUM|RO|WAIT|FINE|GP1)\b/g, '<span class="k">$1</span>')
      .replace(/\b(P\[\d+)/g, '<span class="s">$1</span>');
  }

  function renderKeyframes(frames, seg) {
    filmstrip.innerHTML = "";
    const peak = seg.keyIdx.reduce((b, i) => (seg.energy[i] > seg.energy[b] ? i : b), seg.keyIdx[0]);
    seg.keyIdx.forEach((idx) => {
      const fig = document.createElement("figure");
      if (idx === peak) fig.className = "kf-hot";
      fig.appendChild(frames.thumbs[idx]);
      const cap = document.createElement("figcaption");
      cap.textContent = fmtT(frames.times[idx]);
      fig.appendChild(cap);
      fig.style.cursor = "pointer";
      fig.addEventListener("click", () => (player.currentTime = frames.times[idx]));
      filmstrip.appendChild(fig);
    });
    $("kfInfo").textContent = `${N_KEYFRAMES} of ${frames.times.length} sampled`;
    $("energyInfo").textContent = `${seg.runs.length} motion segment${seg.runs.length === 1 ? "" : "s"} · ${STEP_TEMPLATE.length} steps`;
  }

  // ───────── Canvases ─────────
  function fitCanvas(c) {
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }
  function clearCanvas(c) { const { ctx, w, h } = fitCanvas(c); ctx.clearRect(0, 0, w, h); }

  function drawEnergy(seg) {
    const { ctx, w, h } = fitCanvas(energyCanvas);
    ctx.clearRect(0, 0, w, h);
    const pad = 8;
    if (!seg) {
      ctx.fillStyle = css("--muted");
      ctx.font = `12px ${css("--mono")}`;
      ctx.fillText("frame-difference energy appears after conversion", pad + 4, h / 2 + 4);
      return;
    }
    const n = seg.energy.length;
    const X = (i) => pad + (i / (n - 1)) * (w - 2 * pad);
    const Y = (e) => h - pad - ((e - seg.lo) / ((seg.hi - seg.lo) || 1)) * (h - 2 * pad - 10);
    // active runs
    ctx.fillStyle = css("--cyan") + "22";
    seg.runs.forEach(([a, b]) => ctx.fillRect(X(a), 0, Math.max(2, X(b) - X(a)), h));
    // threshold
    ctx.strokeStyle = css("--line-2");
    ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(pad, Y(seg.thr)); ctx.lineTo(w - pad, Y(seg.thr)); ctx.stroke();
    ctx.setLineDash([]);
    // curve
    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, css("--green")); grad.addColorStop(1, css("--cyan"));
    ctx.strokeStyle = grad; ctx.lineWidth = 2; ctx.lineJoin = "round";
    ctx.beginPath();
    seg.energy.forEach((e, i) => (i ? ctx.lineTo(X(i), Y(e)) : ctx.moveTo(X(i), Y(e))));
    ctx.stroke();
    // step ticks
    ctx.fillStyle = css("--green");
    ctx.font = `600 9.5px ${css("--mono")}`;
    seg.stepTimes.forEach((t, k) => {
      const x = pad + (t / seg.duration) * (w - 2 * pad);
      ctx.fillRect(x - 0.5, 0, 1, 10);
      ctx.fillText(String(k + 1), x + 2, 9);
    });
  }

  function drawOverlay(g, fw, fh) {
    const { ctx, w, h } = fitCanvas(overlay);
    ctx.clearRect(0, 0, w, h);
    // account for object-fit: contain letterboxing
    const ar = player.videoWidth && player.videoHeight ? player.videoWidth / player.videoHeight : fw / fh;
    let vw = w, vh = w / ar;
    if (vh > h) { vh = h; vw = h * ar; }
    const ox = (w - vw) / 2, oy = (h - vh) / 2;
    const mark = (o, color, text) => {
      if (!o.img) return;
      const x = ox + o.img.u * vw, y = oy + o.img.v * vh;
      ctx.strokeStyle = color; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 14, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x - 22, y); ctx.lineTo(x - 16, y); ctx.moveTo(x + 16, y); ctx.lineTo(x + 22, y);
      ctx.moveTo(x, y - 22); ctx.lineTo(x, y - 16); ctx.moveTo(x, y + 16); ctx.lineTo(x, y + 22); ctx.stroke();
      ctx.font = `600 11px ${css("--mono")}`;
      const tw = ctx.measureText(text).width;
      ctx.fillStyle = "rgba(0,0,0,0.7)"; ctx.fillRect(x + 18, y - 28, tw + 10, 18);
      ctx.fillStyle = color; ctx.fillText(text, x + 23, y - 15);
    };
    mark(g.pick, "#76b900", `PICK  ${g.pick.label}`);
    mark(g.place, "#22d3ee", "PLACE  target");
  }

  let topAnim = 0;
  function drawTopView(res) {
    cancelAnimationFrame(topAnim);
    const c = $("topView");
    if (!c.offsetParent) return; // hidden tab
    const draw = (phase) => {
      const { ctx, w, h } = fitCanvas(c);
      ctx.clearRect(0, 0, w, h);
      const pad = 26;
      // map UF0: Y → horizontal, X → vertical (far = up)
      const sx = (y) => pad + ((y - ENVELOPE.y[0]) / (ENVELOPE.y[1] - ENVELOPE.y[0])) * (w - 2 * pad);
      const sy = (x) => h - pad - ((x - ENVELOPE.x[0]) / (ENVELOPE.x[1] - ENVELOPE.x[0])) * (h - 2 * pad);
      ctx.strokeStyle = css("--line-2"); ctx.setLineDash([4, 4]);
      ctx.strokeRect(sx(ENVELOPE.y[0]), sy(ENVELOPE.x[1]), sx(ENVELOPE.y[1]) - sx(ENVELOPE.y[0]), sy(ENVELOPE.x[0]) - sy(ENVELOPE.x[1]));
      ctx.setLineDash([]);
      ctx.fillStyle = css("--muted"); ctx.font = `10.5px ${css("--mono")}`;
      ctx.fillText("UF0 top view · demo envelope", pad, 16);
      ctx.fillText("+Y →", w - pad - 30, h - 8);
      if (!res) { ctx.fillText("path appears after conversion", w / 2 - 90, h / 2); return; }
      const pts = res.plan.waypoints.map((wp) => [sx(wp.pose[1]), sy(wp.pose[0]), wp.pose[6]]);
      ctx.strokeStyle = css("--cyan"); ctx.lineWidth = 2;
      ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
      const label = (x, y, t, col) => { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, 5, 0, 7); ctx.fill(); ctx.fillText(t, x + 8, y - 6); };
      ctx.font = `600 10.5px ${css("--mono")}`;
      label(pts[0][0], pts[0][1], "HOME", css("--text-2"));
      label(pts[2][0], pts[2][1], "PICK", css("--green"));
      label(pts[6][0], pts[6][1], "PLACE", css("--cyan"));
      // moving TCP marker
      const segs = pts.length - 1, u = phase * segs, i = Math.min(segs - 1, Math.floor(u)), f = u - i;
      const x = pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f, y = pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f;
      ctx.fillStyle = pts[i + 1][2] ? css("--amber") : css("--text");
      ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fill();
    };
    if (!res) return draw(0);
    const t0 = performance.now();
    const loop = (now) => { draw(((now - t0) / 6000) % 1); topAnim = requestAnimationFrame(loop); };
    topAnim = requestAnimationFrame(loop);
  }

  // Highlight the step being performed while the video plays.
  player.addEventListener("timeupdate", () => {
    if (!result) return;
    const items = [...$("outSteps").querySelectorAll("li[data-t]")];
    let cur = -1;
    items.forEach((li, k) => { if (player.currentTime >= +li.dataset.t - 0.4) cur = k; });
    items.forEach((li, k) => (li.style.borderColor = k === cur ? css("--green") : ""));
  });

  $("vDownload").addEventListener("click", () => {
    if (!result?.planJson) return;
    const blob = new Blob([JSON.stringify(result.planJson, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${result.planJson.plan_id}.plan.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

  window.addEventListener("resize", () => {
    if (result) { drawEnergy(result.seg); drawOverlay(result.ground, result.frames.width, result.frames.height); }
    else drawEnergy(null);
  });
  reset();
})();
