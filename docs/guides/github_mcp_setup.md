# GitHub MCP Server Setup Guide

**Purpose**: Configure the GitHub MCP server for use with ws3 development.  
**Date**: 2026-07-26  
**Status**: Ready for use  

---

## Installation

### Prerequisites

1. **Python 3.9+** with virtual environment
2. **Deno** — required by GitHub MCP server
3. **GitHub token** with appropriate permissions

### Step 1: Install Deno

```bash
curl -fsSL https://deno.land/install.sh | sh
export DENO_INSTALL="/home/gep/.deno"
export PATH="$DENO_INSTALL/bin:$PATH"
deno --version  # Verify installation
```

### Step 2: Install GitHub MCP Server

```bash
# Install ws3 with GitHub MCP support
pip install ws3[github-mcp]

# Or install just the MCP server
pip install github-mcp-server
```

### Step 3: Verify Installation

```bash
github-mcp-cli --help
# Should show:
# GitHub MCP Server v2.5.4 - Code-First Mode (execute_code only)
# Deno: deno 2.9.4 (stable, release, x86_64-unknown-linux-gnu)
```

---

## Configuration

### GitHub Token

Create a GitHub Personal Access Token (PAT) with the following scopes:

- `repo` — Full control of private repositories
- `read:org` — Read org data
- `read:user` — Read user data

**Create token**: https://github.com/settings/tokens/new

### Environment Variables

Set the following environment variables (add to `~/.bashrc` or `~/.zshrc`):

```bash
export GITHUB_TOKEN="ghp_your_token_here"
export GITHUB_REPO="UBC-FRESH/ws3"  # Optional: default repository
```

Or use a `.env` file in your project root:

```bash
GITHUB_TOKEN=ghp_your_token_here
GITHUB_REPO=UBC-FRESH/ws3
```

---

## Usage

### Check Server Health

```bash
github-mcp-cli health
```

### Clear Token Cache

```bash
github-mcp-cli clear-cache
```

### Verify Deno Installation

```bash
github-mcp-cli check-deno
```

---

## Integration with VS Code

Add to `.vscode/settings.json`:

```json
{
  "github.copilot.chat.streaming": true,
  "github.copilot.chat.codeGeneration.model": "github/gpt-oss-120b",
  "github.copilot.chat.streaming.useSSE": true,
  "github.copilot.chat.streaming.sseUrl": "https://api.githubcopilot.com"
}
```

---

## Troubleshooting

### "Deno not found"

Install Deno following Step 1 above.

### "ModuleNotFoundError: No module named 'src'"

This is a known packaging issue. The cli.py has been patched to use direct imports. If you reinstall the package, you may need to re-apply the patch.

### "Permission denied" for GitHub operations

Verify your GitHub token has the required scopes. Recreate the token if needed.

---

## Next Steps

1. **Wire into agent-workbench** — Configure the GitHub MCP server in your main dev environment
2. **Create issue templates** — Standardize issue body format
3. **Set up GitHub Actions** — Automate issue labeling and assignment
4. **Document agent workflow** — Show agents how to interact with GitHub issues

See `planning/phase6_github_mcp_plan.md` for the full Phase 6 plan.