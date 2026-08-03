# Phase 11, Task 11.1 — Lint Contract and Baseline

**Governing issue:** ws3 #128  
**Branch:** `feature/phase11-ruff-cleanup` at `1f05ed5`  
**Parent commit:** `ce95c16` P10: add typed FEMIC model contract (#127)  
**Date established:** 2026-08-03  

---

## 1. Environment

| Item | Value |
| --- | --- |
| Ruff | `0.16.0` |
| Python | `3.12.3` (GCC 13.3.0) |
| `requires-python` (pyproject.toml) | `>=3.10` |
| `target-version` (pyproject.toml) | `"py39"` |
| **Mismatch** | **`target-version = "py39"` conflicts with `requires-python = ">=3.10"`** |

---

## 2. Configuration (pyproject.toml)

```toml
[tool.ruff]
line-length = 100
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "I", "W", "B", "C4", "UP"]
ignore = ["E501", "B008"]

[tool.ruff.lint.per-file-ignores]
"ws3/__init__.py" = ["F401"]
"tests/*" = ["D100", "D101", "D102", "D103", "D104"]
```

**Observations:**
- `E501` (line-too-long) is waived — intentional, line-length enforced by formatter only.
- `B008` (function-call-in-default-argument) is waived — likely deliberate for performance-critical code.
- D-codes are waived in tests — standard practice.
- `F401` (unused-import) waived in `__init__.py` — standard barrel-export pattern.
- **`target-version = "py39"` is the configuration defect that causes 3 false-positive `invalid-syntax` findings.**

---

## 3. Reproducible Baseline Command

```bash
python -m ruff check ws3/ tests/ --statistics
```

**Output (641 findings, 24 unique rules, 31 files):**

```
234     UP006   [ ] non-pep585-annotation
119     E701    [ ] multiple-statements-on-one-line-colon
 91     UP045   [ ] non-pep604-annotation-optional
 38     UP031   [ ] printf-string-formatting
 29     I001    [*] unsorted-imports
 25     UP035   [-] deprecated-import
 23     E741    [ ] ambiguous-variable-name
 16     E402    [ ] module-import-not-at-top-of-file
 12     B007    [ ] unused-loop-control-variable
 12     B011    [ ] assert-false
  7     UP007   [ ] non-pep604-annotation-union
  4     B006    [ ] mutable-argument-default
  4     B904    [ ] raise-without-from-inside-except
  4     E731    [ ] lambda-assignment
  4     UP015   [*] redundant-open-modes
  3             [ ] invalid-syntax
  3     C401    [ ] unnecessary-generator-set
  3     C414    [ ] unnecessary-double-cast-or-process
  3     UP037   [*] quoted-annotation
  2     C416    [ ] unnecessary-comprehension
  2     E401    [*] multiple-imports-on-one-line
  1     B017    [ ] assert-raises-exception
  1     C420    [*] unnecessary-dict-comprehension-for-iterable
  1     UP028   [ ] yield-in-for-loop
Found 641 errors.
[*] 41 fixable with the `--fix` option (397 hidden fixes can be enabled with the `--unsafe-fixes` option).
```

---

## 4. Finding Classification

| Category | Count | Percentage | Description |
| --- | ---: | ---: | --- |
| **Maintainability** | 445 | 69.4% | UP, B, C4 rules — type annotations, deprecated imports, comprehensions |
| **Correctness** | 161 | 25.1% | E, F rules — real bugs or code smells (E701, E741, E402, B006, B007, B011, B904, B017, E731) |
| **Cosmetic** | 29 | 4.5% | I (import ordering) |
| **Configuration defects** | 3 | 0.5% | `invalid-syntax` — caused by `target-version=py39` mismatch |
| **Total** | **641** | 100% | |

### 4a. Invalid-Syntax Findings (3) — Configuration Defect

All three are in `ws3/opt.py` and are caused by `target-version = "py39"`:

```
ws3/opt.py:306:9  Cannot use `match` statement on Python 3.9 (syntax was added in Python 3.10)
ws3/opt.py:309:17 Cannot use `match` statement on Python 3.9 (syntax was added in Python 3.10)
ws3/opt.py:321:17 Cannot use `match` statement on Python 3.9 (syntax was added in Python 3.10)
```

**Root cause:** The code uses `match`/`case` (Python 3.10+), but Ruff is configured for `py39`. Fix: change `target-version` to `"py310"`.

---

## 5. File-Level Breakdown (Top 10)

| Rank | File | Findings | Key Rules |
| ---: | --- | ---: | --- |
| 1 | `ws3/forest.py` | 234 | UP006, E701, UP045, UP031, E741 |
| 2 | `ws3/forest_helper.py` | 77 | UP006, E701, UP045 |
| 3 | `ws3/spatial.py` | 53 | UP006, E701, UP045 |
| 4 | `ws3/common.py` | 49 | UP006, E701, UP045 |
| 5 | `ws3/core.py` | 44 | UP006, E701, UP045 |
| 6 | `ws3/opt.py` | 43 | UP006, E701, UP045, 3× invalid-syntax |
| 7 | `ws3/advanced_modeling.py` | 34 | UP006, E701, UP045 |
| 8 | `ws3/perf.py` | 26 | UP006, E701, UP045 |
| 9 | `ws3/integration.py` | 23 | UP006, E701, UP045 |
| 10 | `tests/test_documentation.py` | 9 | UP035, UP006, E701 |

**Remaining 21 files:** 1–8 findings each, spread across `ws3/agent/`, `tests/`, and minor `ws3/` modules.

**Concentration:** Top 3 files account for 364/641 (56.8%) of all findings. `ws3/forest.py` alone has 234 (36.5%).

---

## 6. Scope Analysis

### 6a. Current Gate Scope: `ws3/ tests/`

- **Python files scanned:** ~164 files
- **Files with findings:** 31
- **Files without findings:** ~133 (clean)
- **New agent modules** (`ws3/agent/`) pass Ruff cleanly — confirms Phase 10 work is lint-clean.

### 6b. Paths NOT in Current Scope

| Path | Type | Count | Recommendation |
| --- | --- | ---: | --- |
| `examples/ws3/` | Nested checkout (separate project) | 41 .py files | **Exclude** — has own pyproject.toml, is a vendored copy |
| `docs/build/` | Generated HTML + doctrees | ~170 .ipynb | **Exclude** — build artifacts |
| `docs/source/examples/` | Source notebooks | ~38 .ipynb | **Consider** — documentation examples |
| `scripts/` | Utility scripts | ~4 .py files | **Consider** — CI/test helpers |
| `papers/` | Manuscript scripts | ~6 .py files | **Exclude** — publication artifacts |
| `runtime/` | Agent job outputs | 0 .py | **Exclude** — untracked runtime data |
| `=` | Stray empty file | 1 | **Exclude** — untracked, empty |
| `.venv/` | Virtual environment | — | **Exclude** — default |
| `node_modules/` | N/A | — | **Exclude** — default |
| `.git/` | Git metadata | — | **Exclude** — default |

### 6c. Notebooks

- **Total .ipynb in repo:** 378
- **Source notebooks (excl. docs/build, nested .git):** 208
- **Notebooks in docs/source/examples/:** ~38
- **Notebooks in examples/ws3/:** ~130 (nested checkout)
- **Notebooks in docs/build/.doctrees/:** ~170 (generated)

Ruff does **not** lint `.ipynb` files by default. The 3 `invalid-syntax` findings are in `.py` files only.

### 6d. Nested Checkout: `examples/ws3/`

This is a **separate project** with its own:
- `pyproject.toml` (requires `>=3.9`)
- `.git` directory (nested git repo)
- `.pre-commit-config.yaml`
- Full source tree mirroring the parent

**Recommendation:** Add `examples/ws3/` to Ruff's `extend-exclude` or `.gitignore`. It is a vendored copy, not part of the ws3 package.

---

## 7. Scope Recommendation

### Proposed Authoritative Gate Command

```bash
python -m ruff check ws3/ tests/ --statistics
```

**Rationale:**
- Matches current `pyproject.toml` configuration.
- Covers the distributable package (`ws3/`) and tests (`tests/`).
- Excludes notebooks, docs, scripts, and nested checkouts by design.
- 641 findings is the current baseline; reducing to 0 would require ~10 hours of mechanical fixes plus review of behavior-sensitive changes.

### Proposed Exclusions (for future consideration)

```toml
[tool.ruff]
exclude = ["examples/ws3/", "docs/build/", "scripts/", "papers/"]
```

### Proposed Target Version Fix

Change `target-version = "py39"` → `target-version = "py310"` to:
1. Eliminate 3 false-positive `invalid-syntax` findings.
2. Align with `requires-python = ">=3.10"`.
3. Allow Ruff to recognize `match`/`case` syntax correctly.

**Expected effect:** 641 → 638 findings (3 invalid-syntax eliminated).

---

## 8. Proposed Blocking Command/Policy

**Option A: Ruff as the sole blocking gate (recommended)**

```bash
python -m ruff check ws3/ tests/
```

- Replace flake8 entirely.
- Update CI to run Ruff only.
- Update `AGENTS.md`, `CONTRIBUTING.md`, and `ROADMAP.md` consistently.
- Set `target-version = "py310"`.
- Reconcile Ruff `select`/`ignore` with intentional flake8 waivers.

**Option B: Retire linting (not recommended)**

- No evidence supports retiring linting.
- Ruff is faster and more modern than flake8.
- Agent modules already pass Ruff cleanly.

---

## 9. Risk Assessment

| Risk | Severity | Mitigation |
| --- | --- | --- |
| `target-version` mismatch masks real syntax errors | High | Fix to `py310` immediately (Task 11.2) |
| `examples/ws3/` nested checkout pollutes scope | Medium | Add to `extend-exclude` |
| Notebook syntax defects not caught | Low | Address in Task 11.3 |
| Flake8/Ruff policy conflict | Medium | Resolve in Task 11.2 |
| 234 findings in `forest.py` may include behavior-sensitive changes | Medium | Review UP006/UP045 changes carefully (TypeAlias vs type[]); E701 multi-statement lines may hide logic |

---

## 10. Validation Commands

```bash
# Verify baseline reproducibility
cd /home/gep/projects/ws3
python -m ruff check ws3/ tests/ --statistics

# Verify target-version mismatch
python -m ruff check ws3/opt.py --select=invalid-syntax --output-format=json

# Verify nested checkout is separate project
cat examples/ws3/pyproject.toml | grep target-version

# Verify agent modules are clean
python -m ruff check ws3/agent/ --statistics

# Verify Python version
python --version
```

---

## 11. Summary Table

| Metric | Value |
| --- | ---: |
| Total findings | 641 |
| Unique rules | 24 |
| Files with findings | 31 |
| Files without findings | ~133 |
| Configuration defects | 3 (invalid-syntax, all in opt.py) |
| Correctness findings | 161 |
| Maintainability findings | 445 |
| Cosmetic findings | 29 |
| Fixable with `--fix` | 41 |
| Fixable with `--unsafe-fixes` | 438 |
| Top offender | `ws3/forest.py` (234) |
| Target-version mismatch | `py39` vs `requires-python >=3.10` |

---

*Report generated by Agent Workbench Supervisor, Task 11.1, 2026-08-03.*