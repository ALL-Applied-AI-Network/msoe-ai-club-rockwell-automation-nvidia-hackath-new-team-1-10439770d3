# Digital-twin integration guide

This extracted app contains the planning and interaction logic used by the A-Team drag-to-command demonstrator. It is intentionally separated from all physical-workcell credentials and command files.

## Data flow

```text
browser drag → target pose → reach check → waypoint plan → approval state
             → simulated animation or reviewed site adapter → completion update
```

The browser treats the server state as authoritative. In simulation mode it animates the plan. In live mode a site-specific adapter would report observed TCP, gripper, and part state, then mark a move complete only after the robot confirms each waypoint.

## Configuration

The MCP wrapper accepts configuration only from environment variables:

- `TWIN_URL` — local or deployed twin base URL;
- `AGENT_TOKEN` — bearer token issued by the deployment; and
- `AGENT_NAME` — display name for audit events.

Never commit the token or embed a physical endpoint in source. Use a deployment secret store and rotate credentials after a demo.

## Motion plan

For a simple top-down transfer the planner emits:

1. source approach;
2. guarded descent;
3. close;
4. proof lift;
5. high-clearance transfer;
6. guarded lower;
7. release; and
8. retreat.

The analytical arm visualization is a reachability aid, not a collision-certified industrial robot model. A physical adapter must perform installation-specific frame transforms, limits, collision checks, completion waits, and safety interlocks.

## Local development

Use a simulated server and disposable test token. Register the MCP process with `node apps/digital-twin/mcp/twin-mcp.mjs`, then use the state and planning tools against simulation mode. Do not connect the public example directly to a robot.

