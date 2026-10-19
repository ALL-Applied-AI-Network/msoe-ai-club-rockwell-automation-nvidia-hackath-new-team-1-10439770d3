# The A-Team — Access Reference

This runbook records how the team's services fit together without placing live
credentials in Git. Passwords, private keys, bearer tokens, private run tokens,
and team-join codes must remain in the service portal, OS credential store, or a
password manager.

## Quick access map

| Service | Purpose | Safe connection reference | Authentication source |
|---|---|---|---|
| Official GitHub repository | Shared code and evidence | Repository URL in `PROJECT_REFERENCE.md` | Git Credential Manager / GitHub account |
| YAAD Lab portal | Team workspace, compute pool, tokens, keys | `https://www.yaadops.com` | UWM email sign-in code |
| YAAD interactive Spark | Persistent shell and GPU workspace | `ssh yaad-ateam` | Local YAAD-only ED25519 private key |
| YAAD MCP | Capacity and batch-job API | `https://www.yaadops.com/api/mcp` | `YAAD_TOKEN` user environment variable |
| CSI VPN | Required for Spark SSH off the CSI network | FortiClient profile `CSI-RemoteAccess`; gateway `vpn.csi.uwm.edu:443` | CSI/UWM login stored in FortiClient or password manager |
| SIDC evaluation platform | Submit and inspect physical-policy rollouts | `https://hackathon.sidclab.org/` | Event account and per-run official workflow |
| NVIDIA Brev | Secondary GPU compute | NVIDIA Brev account provisioned for captain | NVIDIA/Brev account |

## SSH identity

Current local identity:

```text
Private key: C:\Users\abdou\.ssh\id_ed25519_yaad_ateam
Public key:  C:\Users\abdou\.ssh\id_ed25519_yaad_ateam.pub
Type:        ED25519
Fingerprint: SHA256:kin0q5CSyEosCjh3x3/YBQHFl2XDDg0UREr2jUVhX2g
Comment:     The A-Team YAAD replacement 2026-10-19
```

The public key is safe to paste into YAAD's **Replace my SSH keys** field. The
private key is the same path without `.pub`; never paste, upload, or commit it.

Connection check:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=10 yaad-ateam yaad-diag
```

If it times out, connect the CSI VPN first. If it says `Permission denied
(publickey)`, compare this computer's public-key fingerprint with the key shown
in the YAAD portal.

## YAAD token

The MCP bearer token is stored by **name**, not value, in the Windows user
environment variable:

```text
YAAD_TOKEN
```

It expires after 14 days and can be revoked from **YAAD → Agent access**. The
token used during initial setup was exposed in chat and must be revoked and
replaced. When replacing it, update the environment variable without writing the
value into a project file:

```powershell
[Environment]::SetEnvironmentVariable('YAAD_TOKEN', '<new token>', 'User')
```

Open a new terminal after the update. Never print the variable in logs or use a
command that puts the value into shell history on a shared machine.

## VPN

Known safe profile settings:

```text
Type:           SSL-VPN
Profile:        CSI-RemoteAccess
Remote gateway: vpn.csi.uwm.edu
Port:           443
Certificate:    None unless CSI explicitly changes the instructions
```

The FortiClient `FA_Scheduler` Windows service must be running. Repair commands
require an administrator PowerShell:

```powershell
Set-Service -Name FA_Scheduler -StartupType Automatic
Start-Service -Name FA_Scheduler
Get-Service -Name FA_Scheduler
```

Do not record the VPN password here. The password exposed during setup should be
changed.

## Team codes and private run tokens

- Retrieve the current YAAD team code from the signed-in team workspace and
  share it directly only with intended teammates.
- Retrieve SIDC run tokens only through the official submission/evaluation
  workflow.
- Never give a private run token to an AI-facing MCP tool.
- Never commit `.request.json`, token exports, browser-storage dumps, or console
  screenshots that include credentials.

## Local-only operational card

The repository root may contain an ignored file named
`PRIVATE_ACCESS_REFERENCE.local.md`. It records this computer's raw private-LAN
SSH target and public identity for convenience, but still contains no password,
private key body, bearer-token value, or team-join code. `.gitignore` prevents it
from entering Git.

## Rotation does not delete work

Changing a password, replacing a public SSH key, or revoking a token does not
delete the Git repository, DGX workspace, simulation results, or source files.
It only changes the credential used to reach them.
