# Security policy

## Reporting

Please report a suspected credential, endpoint, privacy, or robot-safety issue privately to the repository maintainers rather than opening a public issue.

## Repository scope

This release must not contain:

- API keys, tokens, passwords, cookies, private keys, or browser profiles;
- internal network addresses or live robot endpoints;
- raw facility recordings without explicit approval;
- files that enable bypass of the official robot-session approval process; or
- unpublished live calibration or motion data presented as portable coordinates.

The public code generates plans for review. It does not authorize motion or replace the robot controller, safety system, site procedures, or qualified operator.

Historical targets in the evidence package are retained to make the reported experiments auditable. They are installation-specific, nonportable, and cannot contact a robot without the separate facility client, credentials, approval process, and on-site safety controls, none of which are included here.

