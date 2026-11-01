# Drag-to-command digital twin

This folder contains the portable planning core extracted from the team's physical-table interface.

## Components

- `client/twin-core.mjs` — geometry, queue, and plan helpers;
- `client/twin-demo.js` — Three.js cell interaction and analytical IK visualization;
- `twin-api.js` — HTTP/API boundary for generated plans;
- `mcp/twin-mcp.mjs` — MCP tool surface;
- `SITE_GUIDE.md` — original operating and architecture guide.

Dragging a part creates a target pose, checks the simplified reachable region, reserves a high-clearance transfer, and emits approach, descend, grasp, lift, transfer, lower, release, and retreat waypoints. A target ghost remains visible until execution is confirmed.

The public interactive deployment is [theateam.dev](https://www.theateam.dev/). This extracted folder is not a standalone production cell controller and intentionally contains no credentials, physical command bridge, or live endpoint configuration.

## Safety

Generated plans are drafts. Connecting them to hardware requires a separately reviewed site adapter, an approved robot session, verified frames, collision review, and operator authorization.

