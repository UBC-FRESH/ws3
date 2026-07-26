# Child issue: migrate core modeling modules to typed interfaces and explicit data contracts

## Roadmap task: P2.2

## Status: planned

## Summary

Begin the incremental migration of the core `ws3` modules to typed interfaces and clearer data contracts, starting with the most central public abstractions.

## Scope

- Identify the core public classes and functions used across the package.
- Introduce type hints, aliases, and explicit return/value contracts for selected modules.
- Refactor the initial module slice to reduce ambiguity around inputs, outputs, and state.
- Preserve runtime behaviour while improving readability and debuggability.

## Acceptance criteria

- At least one foundational module is migrated to explicit type hints and clearer contracts.
- The refactored module remains importable and passes the existing test baseline.
- The change is documented with sufficient context for future contributors.
