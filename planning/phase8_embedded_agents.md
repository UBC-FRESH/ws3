# Phase 8 — Embedded Agent Capabilities

**Purpose**: Give ws3 a first-class, validated, agent-backed interface so that external
coding agents can *operate* the package through a contract-bound surface instead of
guessing at the Python API.

**Date**: 2026-07-29

**Status**: proposed (not yet started)

**Raw motivation**: see [phase8_developer_brainstorm_notes.md](phase8_developer_brainstorm_notes.md)

---

## Context

The UBC-FRESH ecosystem (`agent-workbench`, `freshforge`, `femic`, `fhops`, `ws3`) is
currently orchestrated **top down**: a team of frontier coding agents reads documentation,
infers the API surface, writes Python, and hopes it composed things correctly. Structure,
contracts, and directives (`AGENTS.md`, planning notes, issue bodies, coordinator profiles)
raise the hit rate, but they are *nudges*. The only validator on generated code is "did it
raise."

Phase 8 adds the **bottom up** half: ws3 ships its own agent-backed capabilities. The
external coordinator no longer hand-writes `fm.import_yields_section(...)` from memory — it
calls a capability that ws3 owns, where ws3 controls the prompt, the model endpoint, and
crucially **the validation of the model's output against real ws3 state**.

The reliability gain does not come from embedding an LLM. It comes from the fact that a
component *inside* the package can cheaply check the answer before returning it.

ws3 is the first **adopter**. The reusable machinery lives in a new standalone
package, `fresh-agent-core`, which `femic`, `fhops`, and `freshforge` also depend
on. Maintainer decision, 2026-07-29: extract the contract up front rather than
proving it in ws3 and refactoring later.

---

## Package Split

```
fresh-agent-core/                  ws3/  (and femic, fhops, freshforge)
├── config.py    endpoint/model    └── agent/
├── provider.py  OpenAI client         ├── __init__.py   registry wiring
├── capability.py Capability ABC       └── capabilities/
├── provenance.py JSONL + redaction        ├── build_mask.py
├── errors.py                              ├── explain_exception.py
├── testing.py   FakeProvider              └── diagnose_import.py
└── mcp/server.py generic MCP host
```

**`fresh-agent-core` owns the mechanism**: config resolution, the provider client,
the `Capability` ABC and its validate/retry loop, provenance with secret
redaction, the `FakeProvider` test double, and a generic MCP server that can host
any registry of capabilities.

**Each package owns its capabilities and their validators.** Only ws3 knows how to
check that a mask resolves, and only femic knows what makes a femic answer valid.
The validator is the domain-specific part and cannot be centralized.

This keeps the oracle rule enforceable in one place while letting each package
contribute domain knowledge — and means a fix to the retry loop or a redaction
bug lands once, not four times.

### Dependency direction

`fresh-agent-core` depends on nothing in the ecosystem. Packages depend on it,
never the reverse. It must not import ws3, femic, fhops, or freshforge.

### Versioning

`fresh-agent-core` is versioned independently and pinned with a compatible-release
constraint, e.g. `fresh-agent-core~=0.1`. It stays an optional extra everywhere:
`pip install ws3[agent]`.

---

## Core Architectural Principle

> **A capability is a prompt plus a validator plus a retry budget. No oracle, no capability.**

Every capability runs a closed loop:

```
build prompt → call model → parse → validate against real ws3 state
                   ↑                            │
                   └──── feed failure back ─────┘  (bounded retries)
```

Output that fails validation **never** reaches the caller. The capability returns
`ok=False` with the accumulated errors instead.

Three consequences fall out of this:

1. **A small model is sufficient.** With a narrow task, a hard oracle, and bounded retries,
   the model only needs to emit plausible candidates cheaply. `Ornith-1.0-9B-GGUF:Q4_K_M`
   at `temperature: 0` is correctly sized for this. Frontier reasoning stays where judgment
   is genuinely needed — the coordinator.
2. **Capabilities are advisory, never mutating.** A capability returns a *proposal*. The
   caller applies it. This keeps a nondeterministic component out of the data path of a
   scientific pipeline.
3. **Every attempt is recorded.** Model id, prompt hash, raw output, validation verdict,
   attempt count. Per the repository contract, the log *is* the evidence.

---

## Non-Goals (explicit out-of-scope)

- **No "ask ws3 anything" chat endpoint.** A generic conversational surface relocates
  unreliability inside the package and launders it as an API. If a validator cannot be
  written, the functionality does not become a capability.
- **No hard dependency on a model endpoint.** ws3 is on PyPI. `pip install ws3` must
  continue to work with no network, no credentials, and no agent extras.
- **No code execution of model output.** Capability outputs are data, validated as data.
  Nothing is `eval`'d, `exec`'d, or shelled out.
- **No credentials in the repository.** Endpoint URLs, headers, and secrets come from
  environment or user config only.
- **No silent state mutation.** Capabilities never modify a `ForestModel` in place.
- **Not the LP-constraint drafting capability.** Valuable, but the oracle (build + solve the
  LP) is expensive. Deferred to a later phase.

---

## Architecture

### ws3-side layout

The shared machinery lives in `fresh-agent-core` (see Package Split above). ws3
contributes only its capabilities and the registry wiring:

```
ws3/agent/
    __init__.py          # public surface: available(), get(), list_capabilities()
    capabilities/
        __init__.py      # registry, assembled from fresh_agent_core.Capability
        build_mask.py
        explain_exception.py
        diagnose_import.py
```

`ws3/agent/` is imported lazily. No core ws3 module imports it at module scope, so
`import ws3` performs no network I/O and works without the `agent` extra installed.

### Capability contract

Defined once in `fresh-agent-core` and subclassed per capability:

```python
class Capability(ABC):
    name: str                      # stable identifier
    description: str               # what the external agent reads to decide to call it
    max_attempts: int = 3

    @abstractmethod
    def build_messages(self, inputs, failures) -> list[dict]: ...

    @abstractmethod
    def parse(self, raw: str): ...          # raw completion -> candidate

    @abstractmethod
    def validate(self, candidate, context) -> Verdict: ...   # MUST touch real ws3 state
```

```python
@dataclass(frozen=True)
class CapabilityResult:
    ok: bool
    value: Any | None
    attempts: int
    provenance_ids: list[str]
    errors: list[str]
```

### Configuration

Resolution order, first hit wins:

1. explicit `AgentConfig(...)` argument
2. environment: `WS3_AGENT_ENDPOINT`, `WS3_AGENT_MODEL`, `WS3_AGENT_API_KEY`,
   `WS3_AGENT_HEADERS` (JSON)
3. user config file: `~/.config/ws3/agent.toml`
4. otherwise **unavailable** — `available()` returns `False`, capabilities raise
   `AgentUnavailable` with an actionable message

Reference target for development is the self-hosted `fresh-llm01` OpenAI-compatible
endpoint running `Ornith-1.0-9B-GGUF:Q4_K_M`, but nothing about that endpoint is
hardcoded.

### Provenance record

One record per **attempt**, including failed attempts:

| field | notes |
|---|---|
| `id` | uuid4 |
| `timestamp` | UTC ISO-8601 |
| `capability` | capability name |
| `model` | model id as reported by config |
| `endpoint_host` | host only — never the full URL with credentials |
| `prompt_sha256` | hash, not the prompt body, by default |
| `raw_output` | model completion |
| `verdict` | `ok` / validation failure reasons |
| `attempt` | 1-based |
| `duration_ms` | wall clock |
| `ws3_version` | `ws3.__version__` |

Sink: JSONL at `WS3_AGENT_LOG`, defaulting to `./.ws3/agent-provenance.jsonl`.
All header values and API keys are redacted before write.

---

## Candidate Capabilities

Selected on one criterion: **is there a cheap, real oracle?**

| Capability | Input | Output | Validator (the oracle) |
|---|---|---|---|
| `build_mask` | NL description + `ForestModel` | mask expression | mask resolves against `fm` to ≥1 development type |
| `explain_exception` | exception + traceback + ws3 context | plain-language cause + next actions | every ws3 symbol, attribute, and path referenced in the output actually exists |
| `diagnose_import` | Woodstock model path + failing section | structured diagnosis + suggested fix | re-parsing the named section with the suggestion applied to a scratch copy succeeds |

`explain_exception` is deliberately included: its validator is a symbol-existence check,
which is exactly the defect class Phase 6 spent its entire scope removing from the
documentation. The same check now runs at runtime, automatically.

Deferred to a later phase: constraint drafting for `opt.Problem`, yield-curve plausibility
QA, scenario synthesis.

---

## Task Breakdown

Tasks 8.1 and 8.2 land in the `fresh-agent-core` repository; 8.3 onward land in ws3.

### Task 8.1 — `fresh-agent-core`: runtime foundation

- [ ] New repository and package skeleton, independently versioned
- [ ] `config.py` — `AgentConfig` + resolution order
- [ ] `provider.py` — OpenAI-compatible client with timeouts and transport retries
- [ ] `errors.py` — `AgentUnavailable`, `ProviderError`, `ValidationExhausted`
- [ ] `available()` returns `False` cleanly with no configuration
- [ ] `testing.py` — `FakeProvider` replaying canned responses, including malformed
- [ ] Unit tests, fully offline

### Task 8.2 — `fresh-agent-core`: capability framework and provenance

- [ ] `capability.py` — `Capability` ABC, `CapabilityResult`, `Verdict`
- [ ] Validate/retry loop with bounded attempts and failure feedback into the next prompt
- [ ] `provenance.py` — `ProvenanceRecord`, JSONL sink, secret redaction
- [ ] `mcp/server.py` — generic MCP host over any capability registry
- [ ] Test: invalid model output never escapes the loop
- [ ] Test: a provenance record is written for every attempt, including failures
- [ ] Test: no secret material appears in any written record
- [ ] Published so ws3 can depend on it

### Task 8.3 — ws3: implement three capabilities

- [ ] `build_mask` + validator + tests
- [ ] `explain_exception` + symbol-existence validator + tests
- [ ] `diagnose_import` + re-parse validator + tests
- [ ] Capability registry and `ws3.agent.list_capabilities()`

### Task 8.4 — ws3: MCP wiring

- [ ] Register the ws3 capability registry with the `fresh-agent-core` MCP host
- [ ] Console entry point (e.g. `ws3-agent-mcp`)
- [ ] Smoke test: server starts, lists ≥3 tools, round-trips one call against `FakeProvider`

### Task 8.5 — Discoverability contract

- [ ] `AGENTS.md` section: the supported agent interface is the capability surface, not
      hand-written API calls
- [ ] Documented MCP registration snippet for `agent-workbench`
- [ ] `README.md` pointer

### Task 8.6 — Packaging and documentation

- [ ] `ws3[agent]` extra depending on `fresh-agent-core`
- [ ] Sphinx page under `docs/source/guides/` — configuration, capabilities, provenance,
      and how to add a capability (validator-first)
- [ ] Worked example
- [ ] `CHANGELOG.md` entry

---

## Acceptance Criteria

- `pip install ws3` succeeds with no agent dependencies; `import ws3` performs no network I/O
- `ws3.agent.available()` returns `False` without configuration and raises nothing
- Every capability either returns validated output or `ok=False` — verified by tests that
  inject malformed and plausible-but-wrong model responses
- A provenance record exists for every attempt, including failures
- No secret material appears in provenance output — asserted by test
- MCP server lists at least three tools
- The entire test suite runs offline with no live endpoint

---

## Verification

```bash
python -m pytest
python -m ruff check .
python -m build
sphinx-build -b html docs/source _build/html -W
```

Plus a manual, non-CI integration check against the live `fresh-llm01` endpoint, recorded
in the phase closeout note.

---

## Risks

| Risk | Mitigation |
|---|---|
| Scope creep toward a general chat interface | The oracle rule is a hard gate on new capabilities |
| Endpoint unavailable or slow | Optional extra, explicit `available()`, graceful degradation, timeouts |
| Credential leakage into logs or repo | Config from env/user file only; redaction with a dedicated test |
| Prompt injection via Woodstock files or tracebacks | Validators are the mitigation; output is never executed |
| Nondeterminism inside a scientific pipeline | `temperature: 0`, advisory-only outputs, full provenance |
| Pattern does not generalize to other packages | `fresh-agent-core` is deliberately thin; only the mechanism is shared, validators stay local |
| Premature abstraction in `fresh-agent-core` | Keep it to config, provider, `Capability`, provenance, MCP host, and the test double. Anything domain-shaped belongs in the adopting package |
| Version skew across four dependent packages | Independent versioning with compatible-release pins; `fresh-agent-core` never imports an ecosystem package |

---

## Ecosystem Generalization

The reusable artifact is the contract: **prompt + mandatory validator + bounded
retry + provenance + MCP exposure**. `fresh-agent-core` makes that contract a
dependency rather than a convention, so `femic`, `fhops`, and `freshforge` inherit
the mechanism and supply only their own capabilities and validators. The
`agent-workbench` Agent Hub Coordinator discovers all of them uniformly through
MCP.

That is the "meet in the middle": top-down coordination supplies intent, bottom-up
capabilities supply verdicts.
