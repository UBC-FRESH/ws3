# Phase 12 — Coverage Target: 52% → 65%

## Parent issue

#144

## Status

Active

## Goal

Raise codecov coverage from 52% to 65% by adding targeted tests for under-covered modules.

## Current coverage baseline

```
TOTAL                                          5051   2408    52%
ws3/forest.py                                  1835   1202    34%
ws3/opt.py                                      312    133    57%
ws3/core.py                                     275    106    61%
ws3/integration.py                              166     35    79%
ws3/common.py                                   295    229    22%
ws3/financial.py                                143     76    47%
ws3/spatial.py                                  212    166    22%
ws3/forest_helper.py                            155    129    17%
ws3/advanced_modeling.py                        199    102    49%
```

## Strategy

### Quick wins: `core.py` (61% → 75%+)
- Uncovered: error branches, debug paths, edge cases on Tree/Node
- ~106 uncovered statements, many are `if`/`raise`/debug paths
- Add tests for: invalid node operations, boundary conditions, debug-mode paths

### `opt.py` (57% → 65%+)
- Uncovered: solver error paths (gurobi/pulp/highspy failure modes)
- ~133 uncovered statements
- Add tests for: missing solver fallback, invalid LP data, status code handling

### `integration.py` (79% → 85%+)
- 35 uncovered statements
- Add tests for the uncovered error/edge case paths

### `forest.py` (34% → 45%+)
- 1202 uncovered statements — too many to target exhaustively
- Focus on: most-used code paths (development types, growth, harvest)
- Model-level integration tests rather than unit-test-every-branch

### NOT in scope for this phase
- `spatial.py` (22%) — requires raster/GIS integration infrastructure
- `forest_helper.py` (17%) — GIS helpers, same
- `common.py` (22%) — utilities used indirectly; covered via other module tests
- Going above 65% in this phase

## Acceptance criteria

- Overall: ≥65%
- `core.py`: ≥75%
- `opt.py`: ≥65%
- `integration.py`: ≥85%
- `forest.py`: ≥45%
- All existing tests still pass

## Verification

```bash
python -m pytest tests/ -q
python -m pytest tests/ --cov=ws3 --cov-report=term-missing
```

## Closeout checklist

- [ ] All acceptance criteria met
- [ ] `planning/phase12_coverage_target.md` updated with final coverage numbers
- [ ] `ROADMAP.md` Phase 12 entry updated to complete
- [ ] PR merged to `main`
