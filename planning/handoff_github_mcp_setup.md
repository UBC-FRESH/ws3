# Handoff — GitHub MCP server setup (Windows 11 + VS Code, or Linux + code-server)

Purpose: a reusable, machine-agnostic procedure for giving a VS Code environment working,
authenticated GitHub MCP tools that do not re-prompt for a token every session. Hand the section
below the horizontal rule to a coding agent on the target machine.

Provenance: executed and verified end to end on Linux + code-server (jupyterhub04) on 2026-07-31,
confirmed by live `get_me` and issue-read calls plus process identification. The Windows steps are
the same design translated to Windows paths and have **not** yet been executed. Treat the Linux
path as verified and the Windows path as unverified until the agent completes step 5.

Why this exists: the previous setup used `${input:}` with the hosted HTTP endpoint. It re-prompted
for a PAT every session on headless Linux (no keyring) and was documented as working without
anyone ever invoking a GitHub tool to check. Both problems are addressed below.

Paste everything below this line into a Copilot Chat agent session on the target machine.

---

Set up the GitHub MCP server in my VS Code environment so that authenticated GitHub tools are
available in Copilot Chat agent mode, **without re-prompting me for a token every session**.

First, tell me which environment you have detected — Windows 11 + VS Code Desktop, or Linux +
code-server — and follow the matching column throughout. Do not guess; check the OS and whether
the editor is code-server.

## Approach

Run the official `github-mcp-server` binary over stdio, with the token supplied from a local file
via `envFile`. Do not use `${input:...}` and do not use `${env:...}`. Rationale is in the
constraints section; read it before deviating.

## Step 1 — Install the binary

Get the latest release from https://github.com/github/github-mcp-server/releases

| | Windows 11 | Linux + code-server |
|---|---|---|
| Asset | `github-mcp-server_Windows_x86_64.zip` | `github-mcp-server_Linux_x86_64.tar.gz` |
| Install to | `%LOCALAPPDATA%\github-mcp\github-mcp-server.exe` | `~/.local/bin/github-mcp-server` |
| Post-install | — | `chmod +x` |

Use the `arm64` asset instead if the machine is ARM. Confirm with `github-mcp-server --version`
and show me the output. Do not proceed until that prints a version.

Do not substitute `@modelcontextprotocol/server-github` via npx, a Docker image, or a
`github-mcp-cli` wrapper. If you cannot install the binary, stop and tell me why rather than
falling back to something else.

## Step 2 — Create the token file

| | Windows 11 | Linux + code-server |
|---|---|---|
| Path | `%USERPROFILE%\.config\github-mcp\.env` | `~/.config/github-mcp/.env` |
| Permissions | `icacls`, remove inheritance, grant only current user | `chmod 700` dir, `chmod 600` file |

Contents, exactly one line:

```
GITHUB_PERSONAL_ACCESS_TOKEN=<my token>
```

Create the file with the value **empty** and tell me to paste the token in myself. Do not ask me
for the token in chat, do not echo it, and do not read the file back to me. If I do not already
have a PAT, tell me to create one with `repo` scope, plus `read:org` and `read:user` if I want org
and team queries.

## Step 3 — Write the MCP config

This is a **user-level** config. Do not put it in a workspace `.vscode/mcp.json`.

| Environment | Path |
|---|---|
| Windows 11 | `%APPDATA%\Code\User\mcp.json` (`Code - Insiders` if applicable) |
| Linux + code-server | `~/.local/share/code-server/User/mcp.json` |

On Linux, also check for `~/.vscode-server/data/User/mcp.json` and `~/.config/Code/User/mcp.json`.
If more than one exists, back each up with a timestamped suffix and write the same content to all
of them, then confirm they are identical. Divergent copies across these locations are a real
failure mode and cause confusing behaviour.

Content:

```json
{
  "servers": {
    "github": {
      "type": "stdio",
      "command": "<absolute path to the binary from step 1>",
      "args": ["stdio"],
      "envFile": "${userHome}/.config/github-mcp/.env"
    }
  }
}
```

Forward slashes work on Windows. Back up any existing config before overwriting it.

## Step 4 — Start the server

Command palette -> `MCP: List Servers` -> select `github` -> Start/Restart Server.

## Step 5 — Verification (do not skip, do not substitute)

A well-formed config file is **not** evidence that anything works. Neither is the server appearing
in `MCP: List Servers`. Prove it by invoking tools:

1. Call the GitHub `get_me` tool. Show me the returned `login`.
2. Call a repository read — for example issue 114 in `UBC-FRESH/ws3` — and show me the title.
3. Confirm which process is serving the calls:
   - Linux: `pgrep -af github-mcp-server`
   - Windows: `Get-Process github-mcp-server | Format-List Path`

   This step matters. A previously configured server can still be alive in the session and answer
   your calls, making a broken new config look like it works.

If any call fails, report the exact error text and the MCP server log contents. Do not adjust
config and declare success without re-running all three checks.

Finally, tell me to reload the window and confirm I am **not** prompted for a token. That is the
whole point of the exercise.

## Constraints — do not "improve" on these

- The top-level key is `servers`, **not** `mcpServers`. `mcpServers` is the Claude Desktop format;
  VS Code ignores it silently, so the config looks fine and does nothing.
- Do not use `${input:...}` with `"password": true`. That stores the token in VS Code
  `SecretStorage`, which needs an OS keyring. On a headless Linux box there is typically no
  keyring, no libsecret and no session D-Bus, so it falls back to in-memory storage and
  **re-prompts every session**. `envFile` avoids the issue on both platforms.
- Do not use `${env:...}`. The VS Code process environment differs from the shell environment,
  especially under code-server, and it does not resolve reliably.
- Never commit the `.env` file, print it, or paste its contents anywhere. On Linux, if it is
  inside a git repo, stop and move it out.

## Troubleshooting notes

- Tool names appear as `mcp_github_mcp_se_*`. The `github-mcp-se` fragment is a **truncation of
  `github-mcp-server`**, not a configured server name. Do not grep config files for it.
- `TypeError: Cannot read properties of undefined (reading 'invoke')` on every tool call means a
  stale tool registration: the tools are registered from an earlier session but no server is
  running now. Restart the server; do not edit config.
- To confirm a server genuinely started, look for a per-server MCP log in the session log
  directory (Linux: `~/.local/share/code-server/logs/<session>/`). A lone `mcpGateway.log`
  containing only `Initialized` means no server started.
- `github-mcp-cli` installed in a Python venv is not this server and requires Deno.
