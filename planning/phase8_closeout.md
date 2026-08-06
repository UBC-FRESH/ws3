# Phase 8 — Implementation/Task Closeout

**Date**: 2026-08-06
**Branch**: `feature/ws3-phase8-embedded-agents`
**Status**: implementation closeout (not GitHub parent issue closure)

---

## This is an implementation closeout

This document records that the Phase 8 **task surface** (Tasks 8.1–8.6) has been
shipped and verified. It is **not** a GitHub parent issue closure. The parent
issue [#105](https://github.com/UBC-FRESH/ws3/issues/105) remains active in the
ROADMAP.

Phase 8 is a capability delivery phase, not a lifecycle milestone. Closing it
does not change repository governance, branching policy, or release cadence.
It records that the six capabilities, their tests, the IPython magics, the MCP
host, and the packaging/documentation artifacts are all present and verified.

---

## Delivered surfaces

### Capabilities (6)

| Capability | Behavior |
|---|---|
| `build_mask` | Natural-language mask resolved against live `ForestModel` to ≥1 development type |
| `explain_exception` | Every ws3 symbol cited in the explanation validated for existence |
| `diagnose_import` | Suggested fix applied to a scratch copy; section re-imports |
| `rtfm` | Routes to the correct capability; cited doc URLs return HTTP 200 |
| `ws3_hint` | General guidance; every cited symbol and URL validated |
| `inspect_model` | Read-only metadata snapshot: base year, horizon/periods, period length,
theme/action/dtype counts, total area at period 1 — validated against the
actual in-memory `ForestModel` object |

### Deterministic read-only boundary

`inspect_model` never modifies model state. It returns a structured
`InspectResult` populated by reading attribute values from the live object.
No simulation runs, no file writes, no solver calls. Unsupported fields
(plotting, extended time series, custom area filters) are explicitly flagged
rather than guessed.

### IPython / Jupyter magics

Magics operate on a `ForestModel` named `fm` in scope. Full set:

| Magic | Purpose |
|---|---|
| `%ws3_capabilities` | List all registered capabilities |
| `%ws3_inspect_model` | Metadata snapshot of `fm` |
| `%ws3_hint` | General modelling guidance |
| `%build_mask` | Build a development-type mask |
| `%explain_exception` | Explain a ws3 exception |
| `%diagnose_import` | Diagnose a Woodstock import failure |
| `%rtfm` | Route to the correct capability |

### MCP host

`ws3-agent-mcp` console entry point exposes all six capabilities as MCP tools
with per-capability input schemas. Pinned `mcp>=1.0,<2`.

### Packaging

- `ws3[agent]` and `ws3[agent-mcp]` extras declared in `pyproject.toml`
- `ws3[rv]` extra for PaCal-based probabilistic financial analysis
- Companion package: [fresh-agent-core](https://github.com/UBC-FRESH/fresh-agent-core)

---

## Verification

| Check | Result |
|---|---|
| Phase 8 agent tests (`tests/test_agent*.py`) | **177 passed** |
| `inspect_model` tests (`test_agent_inspect_model.py`) | included in above |
| IPython magic tests (`test_agent_ipython_magics.py`) | included in above |
| Full test suite | **330 passed, 11 skipped** |
| Ruff — touched Phase 8 implementation / test files | clean |
| `examples/agent_capability_example.py` | executes successfully — all six capabilities register and `build_mask`, `explain_exception`, `rtfm` return valid results |
| Six capability imports | `build_mask`, `explain_exception`, `diagnose_import`, `inspect_model`, `rtfm`, `ws3_hint` — all resolve |
| `git diff --check` | clean (no formatting errors) |
| `sphinx-build -q -b dummy docs/source /tmp/ws3-phase8-docs-final` | exited 0, no errors |

### Sphinx caveat

Sphinx 9.1.0, sphinxcontrib-mermaid, nbsphinx, and sphinx_rtd_theme are installed
in `ws3/.venv`. The strict command

```
.venv/bin/sphinx-build -b dummy docs/source /tmp/ws3-phase8-docs-strict-final -W
```

runs and exits 1 because of pre-existing non-cross-reference warnings (277 total):

| Category | Count | Scope |
|---|---|---|
| `unknown document` | 0 | — |
| `undefined label` | 0 | — |
| `reference target not found` | 0 | — |
| Duplicate autodoc object descriptions | 217 | pre-existing across module docs |
| Document not in toctree | 37 | examples, appendices, modules — pre-existing, outside Phase 8 |
| Notebook / docutils inline-markup warnings | 12 | docutils interpreted-text warnings in notebooks |
| Title overline too short | 3 | `guides/limitations-and-boundaries`, `textbook/ch17`, `textbook/ch18` |
| Block quote formatting | 2 | pre-existing |
| Unreadable images | 2 | `examples/images/growth_curve_example.png`, `examples/images/aoi_example.png` |
| Title underline too short | 1 | `guides/agent-capabilities.rst` |
| Explicit markup unindent | 1 | pre-existing |
| Notebook missing section title | 1 | `examples/071_*` notebook |
| Unmatched toctree glob `examples/*.ipynb` | 1 | pre-existing |

No strict-build warning points to `docs/source/guides/agent-capabilities.rst`
(other than the title-underline-too-short issue which is outside Phase 8 scope),
and the non-warning build exits 0.

> **Non-warning build result**: `sphinx-build -q -b dummy docs/source
> /tmp/ws3-phase8-docs-final` exited 0 with no errors, confirming the guide
> does not break the doc tree. Strict `-W` now runs apart from the pre-existing
> non-cross-reference warnings listed above.

---

## Out of scope (explicitly not shipped)

- Arbitrary plotting / figure generation
- Extended time-series queries against model history
- Custom area filters beyond the period-1 total area
- mypy typing debt remediation (deferred — [#98](https://github.com/UBC-FRESH/ws3/issues/98))
- MCP 2.x migration (deferred — [fresh-agent-core#1](https://github.com/UBC-FRESH/fresh-agent-core/issues/1))

---

## Documentation mutations in this closeout turn

| File | Change |
|---|---|
| `README.md` | Expanded agent capability table from 3 to 6 entries; added IPython magic usage snippet |
| `ROADMAP.md` | Marked Tasks 8.5 and 8.6 complete; added capability inventory table and test coverage note |
| `CHANGELOG.md` | Unchanged — entries already consistent with six-capability surface |
| `planning/phase8_closeout.md` | Created — this file |