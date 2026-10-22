/*
 * Passcode-gated static server for the A-Team site.
 *
 * Every request (pages, scripts, video, CAD files) is refused until the
 * visitor enters the passcode. A correct passcode sets a signed, HttpOnly
 * cookie valid for 30 days. Set SITE_PASSCODE (and optionally SITE_SECRET)
 * in Railway to change it without a code change.
 */
"use strict";

const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");

const PORT = Number(process.env.PORT) || 8080;
const PASSCODE = process.env.SITE_PASSCODE || "winners";
const SECRET = process.env.SITE_SECRET || crypto.createHash("sha256").update(`ateam:${PASSCODE}`).digest("hex");
const ROOT = path.join(__dirname, "public");
const COOKIE = "ateam_access";
const MAX_AGE_S = 30 * 24 * 3600;
const ATTEMPTS_PER_MIN = 8;

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".ico": "image/x-icon",
  ".mp4": "video/mp4",
  ".webm": "video/webm",
  ".stl": "model/stl",
  ".woff2": "font/woff2",
};

// ───────── Session token: "<expiry>.<hmac>" ─────────
function sign(exp) {
  return crypto.createHmac("sha256", SECRET).update(String(exp)).digest("base64url");
}
function makeToken() {
  const exp = Math.floor(Date.now() / 1000) + MAX_AGE_S;
  return `${exp}.${sign(exp)}`;
}
function validToken(token) {
  const [exp, mac] = String(token || "").split(".");
  if (!exp || !mac || Number(exp) < Date.now() / 1000) return false;
  const expected = Buffer.from(sign(exp));
  const given = Buffer.from(mac);
  return given.length === expected.length && crypto.timingSafeEqual(given, expected);
}
function passcodeMatches(input) {
  const a = crypto.createHash("sha256").update(String(input).trim().toLowerCase()).digest();
  const b = crypto.createHash("sha256").update(PASSCODE.toLowerCase()).digest();
  return crypto.timingSafeEqual(a, b);
}
function readCookie(req, name) {
  for (const part of (req.headers.cookie || "").split(";")) {
    const [k, ...v] = part.trim().split("=");
    if (k === name) return decodeURIComponent(v.join("="));
  }
  return null;
}

// ───────── Brute-force throttle (per client IP) ─────────
const attempts = new Map();
function clientIp(req) {
  return (req.headers["x-forwarded-for"] || "").split(",")[0].trim() || req.socket.remoteAddress;
}
function throttled(ip) {
  const now = Date.now();
  const list = (attempts.get(ip) || []).filter((t) => now - t < 60_000);
  list.push(now);
  attempts.set(ip, list);
  return list.length > ATTEMPTS_PER_MIN;
}
setInterval(() => {
  const now = Date.now();
  for (const [ip, list] of attempts) if (list.every((t) => now - t >= 60_000)) attempts.delete(ip);
}, 60_000).unref();

// ───────── Responses ─────────
const SECURITY_HEADERS = {
  "X-Content-Type-Options": "nosniff",
  "Referrer-Policy": "same-origin",
  "X-Frame-Options": "SAMEORIGIN",
};

function safeNext(next) {
  // Same-origin paths only: "//host" and "/\host" are treated as other origins by browsers.
  return typeof next === "string" && /^\/(?![\/\\])/.test(next) && !next.includes("\\") ? next : "/";
}

function loginPage(res, { next = "/", error = "", status = 401 } = {}) {
  const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  res.writeHead(status, { ...SECURITY_HEADERS, "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" });
  res.end(`<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="robots" content="noindex" />
<title>The A-Team · Access</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..125,400..800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
<style>
  :root { --bg:#070d17; --panel:#0f1b2b; --line:#1f3149; --line-2:#2a4262; --text:#e6edf6; --text-2:#9fb0c6; --muted:#6c7f98; --green:#76b900; --green-ink:#0a1400; --cyan:#22d3ee; --violet:#a855f7; --red:#f36b6b; color-scheme: dark; }
  @media (prefers-color-scheme: light) { :root { --bg:#f4f6f9; --panel:#fff; --line:#dbe2ec; --line-2:#c4cfdd; --text:#0d1726; --text-2:#43546b; --muted:#6b7a8f; --green:#4f8a00; --green-ink:#fff; --cyan:#0891b2; --violet:#7c3aed; --red:#c53030; color-scheme: light; } }
  * { box-sizing: border-box; }
  body { margin:0; min-height:100vh; display:grid; place-items:center; padding:16px; background:var(--bg); color:var(--text); font-family:"Archivo",system-ui,sans-serif; }
  body::before { content:""; position:fixed; inset:0; pointer-events:none; opacity:.5;
    background-image:linear-gradient(var(--line) 1px,transparent 1px),linear-gradient(90deg,var(--line) 1px,transparent 1px); background-size:44px 44px;
    mask-image:radial-gradient(ellipse 70% 60% at 50% 40%,#000 10%,transparent 75%); }
  .card { position:relative; width:100%; max-width:420px; background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:32px 28px; box-shadow:0 30px 60px -30px rgba(0,0,0,.6); }
  .eyebrow { font:500 11.5px "JetBrains Mono",monospace; letter-spacing:.06em; text-transform:uppercase; color:var(--green); margin:0 0 14px; }
  h1 { margin:0 0 8px; font-size:52px; line-height:.9; font-weight:800; font-stretch:118%; letter-spacing:-.045em; }
  h1 span { background:linear-gradient(90deg,var(--green),var(--cyan) 60%,var(--violet)); -webkit-background-clip:text; background-clip:text; color:transparent; }
  p { margin:0 0 22px; color:var(--text-2); font-size:15px; }
  label { display:block; font-size:13px; font-weight:600; margin-bottom:8px; }
  input { width:100%; font:500 16px "JetBrains Mono",monospace; padding:13px 14px; border-radius:10px; border:1px solid var(--line-2); background:var(--bg); color:var(--text); }
  input:focus { outline:2px solid var(--cyan); outline-offset:1px; }
  button { width:100%; margin-top:14px; font:600 15px "Archivo",sans-serif; padding:13px; border:0; border-radius:10px; background:var(--green); color:var(--green-ink); cursor:pointer; }
  button:hover { filter:brightness(1.08); }
  .err { color:var(--red); font-size:13.5px; margin:10px 0 0; min-height:1em; }
  .foot { margin-top:20px; font:11.5px "JetBrains Mono",monospace; color:var(--muted); }
</style>
</head>
<body>
  <form class="card" method="post" action="/__access">
    <p class="eyebrow">Rockwell Automation × NVIDIA Innovation Lab</p>
    <h1>The <span>A-Team</span></h1>
    <p>This site is private. Enter the team passcode to continue.</p>
    <label for="code">Passcode</label>
    <input id="code" name="passcode" type="password" autocomplete="current-password" autofocus required />
    <input type="hidden" name="next" value="${esc(next)}" />
    <button type="submit">Enter</button>
    <div class="err" role="alert">${esc(error)}</div>
    <div class="foot">Innovation Lab · Fall 2026</div>
  </form>
</body>
</html>`);
}

function serveFile(req, res, urlPath) {
  let rel;
  try { rel = decodeURIComponent(urlPath); } catch { rel = urlPath; }
  if (rel.endsWith("/")) rel += "index.html";
  const file = path.normalize(path.join(ROOT, rel));
  if (!file.startsWith(ROOT + path.sep)) { res.writeHead(403, SECURITY_HEADERS); return res.end("Forbidden"); }

  fs.stat(file, (err, st) => {
    if (err || !st.isFile()) {
      res.writeHead(404, { ...SECURITY_HEADERS, "Content-Type": "text/plain; charset=utf-8" });
      return res.end("Not found");
    }
    const type = TYPES[path.extname(file).toLowerCase()] || "application/octet-stream";
    const headers = {
      ...SECURITY_HEADERS,
      "Content-Type": type,
      "Accept-Ranges": "bytes",
      // Private: gated content must not land in shared caches.
      "Cache-Control": type.startsWith("text/html") ? "private, no-cache" : "private, max-age=3600",
      "Last-Modified": st.mtime.toUTCString(),
    };
    // Range support so <video> can seek (required by Safari).
    const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range || "");
    if (range) {
      let start = range[1] ? Number(range[1]) : st.size - Number(range[2]);
      let end = range[1] && range[2] ? Number(range[2]) : st.size - 1;
      if (!(start >= 0 && start <= end && end < st.size)) {
        res.writeHead(416, { ...headers, "Content-Range": `bytes */${st.size}` });
        return res.end();
      }
      res.writeHead(206, { ...headers, "Content-Range": `bytes ${start}-${end}/${st.size}`, "Content-Length": end - start + 1 });
      if (req.method === "HEAD") return res.end();
      return fs.createReadStream(file, { start, end }).pipe(res);
    }
    res.writeHead(200, { ...headers, "Content-Length": st.size });
    if (req.method === "HEAD") return res.end();
    fs.createReadStream(file).pipe(res);
  });
}

// ───────── Router ─────────
const server = http.createServer((req, res) => {
  const url = new URL(req.url, "http://localhost");
  const secure = req.headers["x-forwarded-proto"] === "https";

  if (url.pathname === "/__health") { res.writeHead(200); return res.end("ok"); }

  if (url.pathname === "/__access" && req.method === "POST") {
    let body = "";
    req.on("data", (c) => { body += c; if (body.length > 4096) req.destroy(); });
    req.on("end", () => {
      const form = new URLSearchParams(body);
      const next = safeNext(form.get("next"));
      if (throttled(clientIp(req))) return loginPage(res, { next, error: "Too many attempts. Wait a minute and try again.", status: 429 });
      if (!passcodeMatches(form.get("passcode") || "")) return loginPage(res, { next, error: "Wrong passcode." });
      res.writeHead(303, {
        ...SECURITY_HEADERS,
        Location: next,
        "Set-Cookie": `${COOKIE}=${makeToken()}; Max-Age=${MAX_AGE_S}; Path=/; HttpOnly; SameSite=Lax${secure ? "; Secure" : ""}`,
        "Cache-Control": "no-store",
      });
      res.end();
    });
    return;
  }

  if (url.pathname === "/__logout") {
    res.writeHead(303, { ...SECURITY_HEADERS, Location: "/", "Set-Cookie": `${COOKIE}=; Max-Age=0; Path=/; HttpOnly; SameSite=Lax${secure ? "; Secure" : ""}` });
    return res.end();
  }

  if (!validToken(readCookie(req, COOKIE))) {
    if (req.method !== "GET" && req.method !== "HEAD") { res.writeHead(401, SECURITY_HEADERS); return res.end(); }
    return loginPage(res, { next: url.pathname + url.search });
  }

  if (req.method !== "GET" && req.method !== "HEAD") { res.writeHead(405, SECURITY_HEADERS); return res.end(); }
  serveFile(req, res, url.pathname);
});

server.listen(PORT, () => console.log(`A-Team site on :${PORT} (passcode ${process.env.SITE_PASSCODE ? "from SITE_PASSCODE" : "default"})`));
