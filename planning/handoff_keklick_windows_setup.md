# Handoff: Set Up keklick-copilot Custom Model Providers in VS Code (Windows 11)

## Context

You are setting up the **keklick-copilot** VS Code extension with six custom open-weight model
endpoints (three providers: `fresh-llm01`, `fresh-ollama`, `fresh-vllm`) to mirror the
configuration running on a remote Linux code-server environment.

---

## Prerequisites

- VS Code installed on Windows 11
- GitHub Copilot extension installed and signed in (the keklick extension piggybacks on it)
- Network access to `*.01101.dev` endpoints (VPN or direct — confirm this works first)

---

## Step 1 — Install the keklick-copilot extension

1. Open VS Code.
2. Open the Extensions panel (`Ctrl+Shift+X`).
3. Search for **`keklick1337.keklick-copilot`** ("Keklick Copilot Custom Endpoints").
4. Click **Install**. Minimum required version: **1.0.22**.

Alternatively, from the terminal:
```
code --install-extension keklick1337.keklick-copilot
```

---

## Step 2 — Open User Settings JSON

1. Open the Command Palette (`Ctrl+Shift+P`).
2. Type **`Open User Settings (JSON)`** and select it.
3. This opens `%APPDATA%\Code\User\settings.json`
   (full path: `C:\Users\<your-username>\AppData\Roaming\Code\User\settings.json`).

---

## Step 3 — Add the custom model configuration

Inside the top-level JSON object in `settings.json`, add the following keys. If the file already
has other settings, add these alongside them (do **not** replace existing keys).

```json
"customcopilot.models": [
    {
        "owned_by": "fresh-ollama",
        "id": "ornith:35b-q4_K_M",
        "object": "model",
        "created": 1784823893,
        "configId": "ollama-ornith-35b-native",
        "displayName": "ornith:35b-q4_K_M (Custom Copilot / native Ollama API)",
        "baseUrl": "https://fresh01-ollama.01101.dev",
        "apiMode": "ollama",
        "userAgent": "agent-workbench-worker/1.0",
        "context_length": 131072,
        "max_tokens": 4096,
        "vision": false,
        "tool_calling": true,
        "headers": {
            "CF-Access-Client-Id": "1ddd134da227dc4eaa8d13811d715c4a.access",
            "CF-Access-Client-Secret": "a54d9eeb89b84539a3868579cbc5f11230f0080c7e41f752625bba51298496bf",
            "User-Agent": "agent-workbench-worker/1.0"
        }
    },
    {
        "id": "ornith:9b-q4_K_M",
        "configId": "ollama-ornith-9b-native",
        "owned_by": "fresh-ollama",
        "displayName": "ornith:9b-q4_K_M (Custom Copilot / native Ollama API)",
        "vision": false,
        "tool_calling": true,
        "temperature": 0,
        "baseUrl": "https://fresh01-ollama.01101.dev",
        "apiMode": "ollama",
        "userAgent": "agent-workbench-worker/1.0",
        "context_length": 131072,
        "max_tokens": 4096,
        "headers": {
            "CF-Access-Client-Id": "1ddd134da227dc4eaa8d13811d715c4a.access",
            "CF-Access-Client-Secret": "a54d9eeb89b84539a3868579cbc5f11230f0080c7e41f752625bba51298496bf",
            "User-Agent": "agent-workbench-worker/1.0"
        }
    },
    {
        "id": "qwen3.6-27b-nvfp4",
        "owned_by": "fresh-vllm",
        "displayName": "Qwen 3.6 27B NVFP4",
        "configId": "fresh-vllm-agent",
        "context_length": 250000,
        "max_tokens": 4096,
        "vision": false,
        "tool_calling": true,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.02,
        "reasoning_effort": "high",
        "enable_thinking": true,
        "include_reasoning_in_request": true,
        "thinking": { "type": "enabled" },
        "extra": {
            "chat_template_kwargs": { "enable_thinking": true }
        },
        "baseUrl": "https://fresh01-vllm.01101.dev/v1",
        "apiMode": "openai",
        "userAgent": "agent-workbench-worker/1.0",
        "headers": {
            "CF-Access-Client-Id": "1ddd134da227dc4eaa8d13811d715c4a.access",
            "CF-Access-Client-Secret": "a54d9eeb89b84539a3868579cbc5f11230f0080c7e41f752625bba51298496bf"
        }
    },
    {
        "id": "hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF:Q4_K_M",
        "owned_by": "fresh-llm01",
        "displayName": "Ornith 1.0 9B GGUF Q4_K_M",
        "configId": "fresh-llm01-agent",
        "context_length": 64000,
        "max_tokens": 4096,
        "vision": false,
        "tool_calling": true,
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.02,
        "enable_thinking": false,
        "include_reasoning_in_request": false,
        "thinking": { "type": "disabled" },
        "baseUrl": "https://fresh-llm01.01101.dev/v1",
        "apiMode": "openai",
        "userAgent": "agent-workbench-worker/1.0",
        "headers": {
            "CF-Access-Client-Id": "1ddd134da227dc4eaa8d13811d715c4a.access",
            "CF-Access-Client-Secret": "a54d9eeb89b84539a3868579cbc5f11230f0080c7e41f752625bba51298496bf"
        }
    },
    {
        "id": "hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF:Q5_K_M",
        "owned_by": "fresh-llm01",
        "displayName": "Ornith-1.0-9B-GGUF:Q5_K_M",
        "context_length": 64000,
        "max_tokens": 4096,
        "vision": false,
        "tool_calling": true,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.02,
        "reasoning_effort": "medium",
        "enable_thinking": true,
        "include_reasoning_in_request": true,
        "thinking": { "type": "enabled" },
        "extra": {
            "chat_template_kwargs": { "enable_thinking": true }
        },
        "baseUrl": "https://fresh-llm01.01101.dev/v1",
        "apiMode": "openai",
        "userAgent": "agent-workbench-worker/1.0",
        "headers": {
            "CF-Access-Client-Id": "1ddd134da227dc4eaa8d13811d715c4a.access",
            "CF-Access-Client-Secret": "a54d9eeb89b84539a3868579cbc5f11230f0080c7e41f752625bba51298496bf"
        }
    },
    {
        "id": "hf.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M",
        "owned_by": "fresh-llm01",
        "displayName": "Qwen2.5-Coder-7B-Instruct",
        "context_length": 64000,
        "max_tokens": 4096,
        "vision": false,
        "tool_calling": true,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.02,
        "enable_thinking": false,
        "include_reasoning_in_request": true,
        "thinking": { "type": "disabled" },
        "extra": {
            "chat_template_kwargs": { "enable_thinking": true }
        },
        "baseUrl": "https://fresh-llm01.01101.dev/v1",
        "apiMode": "openai",
        "userAgent": "agent-workbench-worker/1.0",
        "headers": {
            "CF-Access-Client-Id": "1ddd134da227dc4eaa8d13811d715c4a.access",
            "CF-Access-Client-Secret": "a54d9eeb89b84539a3868579cbc5f11230f0080c7e41f752625bba51298496bf"
        }
    }
],
"chat.planAgent.defaultModel": "Qwen 3.6 27B NVFP4 (copilotcustommodelsendpoint)",
"chat.utilitySmallModel": "customendpoint/qwen3.6-27b-nvfp4",
"chat.utilityModel": "customendpoint/qwen3.6-27b-nvfp4",
"inlineChat.defaultModel": "Qwen 3.6 27B NVFP4 (copilotcustommodelsendpoint)",
"chat.byokUtilityModelDefault": "mainAgent",
"customcopilot.debugRequestLogging": true
```

**Important:** The resulting `settings.json` must be valid JSON. After pasting, verify:
- No trailing commas after the last key in any object.
- All braces and brackets are balanced.
- Use `Ctrl+Shift+P` → **Format Document** to auto-format and surface syntax errors.

---

## Step 4 — Reload VS Code

Run from the Command Palette (`Ctrl+Shift+P`):
```
Developer: Reload Window
```

---

## Step 5 — Verify the models appear

1. Open a Copilot Chat panel (`Ctrl+Alt+I`).
2. Click the model picker (the model name shown near the chat input).
3. You should see all six models listed under their respective provider labels:
   - `ornith:35b-q4_K_M (Custom Copilot / native Ollama API)`
   - `ornith:9b-q4_K_M (Custom Copilot / native Ollama API)`
   - `Qwen 3.6 27B NVFP4`
   - `Ornith 1.0 9B GGUF Q4_K_M`
   - `Ornith-1.0-9B-GGUF:Q5_K_M`
   - `Qwen2.5-Coder-7B-Instruct`

---

## Step 6 — Smoke-test a model

Select **Qwen 3.6 27B NVFP4** in the model picker and send a simple message like `hello`. You
should get a response. If you get a connection error, check:

1. Can you reach `https://fresh01-vllm.01101.dev` from this machine's browser?
   If not, the Cloudflare Access tunnel may require additional network access (check with the
   infrastructure owner — the CF service token in the headers handles auth, but the host must
   be reachable).
2. Check the keklick extension output log:
   `View` → `Output` → select **Keklick Copilot** in the dropdown.
   `customcopilot.debugRequestLogging: true` is already set, so request/response details will
   appear there.

---

## Reference: Provider / endpoint map

| `owned_by` label | Endpoint base URL | API mode |
|---|---|---|
| `fresh-ollama` | `https://fresh01-ollama.01101.dev` | `ollama` |
| `fresh-vllm` | `https://fresh01-vllm.01101.dev/v1` | `openai` |
| `fresh-llm01` | `https://fresh-llm01.01101.dev/v1` | `openai` |

All three endpoints are protected by Cloudflare Access. Authentication is handled automatically
via the `CF-Access-Client-Id` and `CF-Access-Client-Secret` request headers already embedded in
each model entry above — no additional login or token is required on the client side.
