"""
Deterministic emission of Woodstock section files from a ModelSpec.

This module provides :py:func:`emit_landscape`, :py:func:`emit_areas`, and
:py:func:`emit_yields`, which emit Woodstock-format section files from a
:py:class:`ws3.agent.spec.ModelSpec`. The emission is deterministic: the same
spec always produces the same bytes, in the same order, with the same
whitespace.

The emitted syntax matches what the existing ws3 importers expect:

- :py:meth:`ws3.forest.ForestModel.import_landscape_section`
- :py:meth:`ws3.forest.ForestModel.import_areas_section`
- :py:meth:`ws3.forest.ForestModel.import_yields_section`

The emitter does not import the model. It only writes text files. Import is
handled by :py:mod:`ws3.agent.builder`.

Period handling: all ages in the emitted files are in *periods* (as Woodstock
expects). The caller is responsible for converting years to periods if needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ws3.agent.spec import ModelSpec


def _normalize_points(points: dict[str, Any]) -> tuple[dict[int, dict[str, float]], int]:
    """
    Normalize yield curve points to a common age-indexed format.

    Accepts two input shapes for each component:
      - Bare numeric sequence: ``[v0, v1, v2, ...]`` — implicit ages
        1, 2, 3, ...
      - Explicit ``(age, value)`` pair sequence: ``[(a0, v0), (a1, v1), ...]``

    Returns:
        - A mapping ``{age: {component_name: value, ...}}`` for all ages
          present across all components.
        - The number of distinct ages (for iteration).

    Raises:
        ValueError: If a component has mixed bare/pair data or if pair
            ages are not strictly increasing.
    """
    # First pass: classify each component's point format.
    component_formats: dict[str, str] = {}  # 'bare' or 'pairs'
    for yname, pts in points.items():
        if not pts:
            component_formats[yname] = 'empty'
            continue
        first = pts[0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            # Could be a pair — verify the first element is numeric.
            if isinstance(first[0], (int, float)):
                component_formats[yname] = 'pairs'
            else:
                # e.g. first element is a string — treat as bare list of
                # non-numeric items (shouldn't happen for 'a'/'t' yields).
                component_formats[yname] = 'bare'
        elif isinstance(first, (int, float)):
            component_formats[yname] = 'bare'
        else:
            raise ValueError(
                f'Yield component {yname!r}: unsupported point format '
                f'(expected numeric or (age, value) pair, got {type(first).__name__})'
            )

    # Check for mixed formats within the same yield.
    formats = {f for f in component_formats.values() if f != 'empty'}
    if len(formats) > 1:
        raise ValueError(
            f'Yield has mixed point formats: {dict(component_formats)}. '
            f'All components must use the same format (all bare values or '
            f'all (age, value) pairs).'
        )

    if 'bare' in formats:
        # Build age-indexed dict with implicit sequential ages (1-based).
        result: dict[int, dict[str, float]] = {}
        max_len = max(len(points.get(n, [])) for n in points)
        for age_idx in range(1, max_len + 1):
            age = age_idx
            row: dict[str, float] = {}
            for yname in points:
                vals = points[yname]
                if age_idx <= len(vals):
                    row[yname] = float(vals[age_idx - 1])
                else:
                    row[yname] = 0.0
            result[age] = row
        return result, len(result)

    # 'pairs' format: extract ages, validate alignment, build result.
    component_ages: dict[str, list[int]] = {}
    for yname, pts in points.items():
        ages: list[int] = []
        for i, item in enumerate(pts):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(
                    f'Yield component {yname!r}: pair at index {i} is not '
                    f'a (age, value) tuple, got {item!r}'
                )
            age_val, val = item
            if not isinstance(age_val, (int, float)):
                raise ValueError(
                    f'Yield component {yname!r}: age at index {i} is not '
                    f'numeric, got {age_val!r}'
                )
            ages.append(int(age_val))
        if ages != sorted(ages) or len(ages) != len(set(ages)):
            raise ValueError(
                f'Yield component {yname!r}: ages must be strictly '
                f'increasing, got {ages}'
            )
        component_ages[yname] = ages

    # Validate all components share the same age sequence.
    ref_ages = component_ages[list(component_ages.keys())[0]]
    for yname, ages in component_ages.items():
        if ages != ref_ages:
            raise ValueError(
                f'Yield component {yname!r} has ages {ages}, expected '
                f'{ref_ages} (all components must share identical ages)'
            )

    # Build the result.
    result = {}
    for age in ref_ages:
        row: dict[str, float] = {}
        for yname, ages in component_ages.items():
            idx = ages.index(age)
            row[yname] = float(points[yname][idx][1])
        result[age] = row
    return result, len(result)


def emit_landscape(spec: ModelSpec, output_dir: Path) -> Path:
    """
    Emit the LANDSCAPE section file.

    :param spec: The model specification.
    :param output_dir: Directory to write the file to.
    :return: Path to the written file.
    """
    path = output_dir / f'{spec.model_name}.lan'
    lines = []
    for theme in spec.themes:
        lines.append(f'*THEME {theme.description}')
        for code in theme.basecodes:
            lines.append(code)
        for agg_name, agg_members in theme.aggregates.items():
            lines.append(f'*AGGREGATE {agg_name}')
            for member in agg_members:
                lines.append(member)
    path.write_text('\n'.join(lines) + '\n')
    return path


def emit_areas(spec: ModelSpec, output_dir: Path) -> Path:
    """
    Emit the AREAS section file.

    Development types are emitted in sorted key order. Ages within each
    development type are emitted in sorted order.

    :param spec: The model specification.
    :param output_dir: Directory to write the file to.
    :return: Path to the written file.
    """
    path = output_dir / f'{spec.model_name}.are'
    lines = []
    for key in sorted(spec.areas.keys()):
        ages = sorted(spec.areas[key].keys())
        for age in ages:
            area = spec.areas[key][age]
            if area < spec.area_epsilon:
                continue
            # Woodstock format: *A <theme1> <theme2> ... <age> <area>
            parts = [str(c) for c in key] + [str(age), f'{area:.6f}']
            lines.append(f'*A {" ".join(parts)}')
    path.write_text('\n'.join(lines) + '\n' if lines else '')
    return path


def emit_yields(spec: ModelSpec, output_dir: Path) -> Path:
    """
    Emit the YIELDS section file.

    Yields are emitted in the order they appear in the spec. Each yield
    definition starts with ``*Y`` followed by the mask, then ``_AGE`` (for
    age-based yields) or the curve expressions (for complex yields).

    :param spec: The model specification.
    :param output_dir: Directory to write the file to.
    :return: Path to the written file.
    """
    path = output_dir / f'{spec.model_name}.yld'
    lines = []
    for yield_spec in spec.yields:
        mask_str = ' '.join(yield_spec.mask)
        lines.append(f'*Y {mask_str}')
        if yield_spec.ytype in ('a', 't'):
            # Age-based or time-based yield: _AGE followed by component
            # names, then points. Points may be bare numeric sequences
            # (implicit sequential ages 1..N) or explicit (age, value)
            # pair sequences.
            lines.append(f'_AGE {" ".join(yield_spec.ynames)}')
            try:
                age_indexed, n_ages = _normalize_points(yield_spec.points)
            except ValueError as exc:
                raise ValueError(
                    f'Yield {yield_spec.mask}: {exc}'
                ) from exc
            for age in sorted(age_indexed.keys()):
                row = age_indexed[age]
                point_parts = [str(age)]
                for yname in yield_spec.ynames:
                    point_parts.append(f'{row.get(yname, 0.0):.6f}')
                lines.append(' '.join(point_parts))
        elif yield_spec.ytype == 'complex':
            # Complex yield: each component is an expression string.
            for yname in yield_spec.ynames:
                expr = yield_spec.points.get(yname, '0')
                lines.append(f'{yname} {expr}')
    path.write_text('\n'.join(lines) + '\n' if lines else '')
    return path


def emit_actions(spec: ModelSpec, output_dir: Path) -> Path:
    """
    Emit the ACTIONS section file.

    Actions are emitted in the order they appear in the spec (dict insertion
    order). Each action starts with ``*ACTION {acode} Y`` (age-independent)
    or ``*ACTION {acode}`` (age-dependent), followed by ``*OPERABLE {acode}``
    and one operable mask per entry with age constraints.

    :param spec: The model specification.
    :param output_dir: Directory to write the file to.
    :return: Path to the written file.
    """
    path = output_dir / f'{spec.model_name}.act'
    lines = ['ACTIONS']
    for acode, action in spec.actions.items():
        lines.append(f'*ACTION {acode} Y')
        lines.append(f'*OPERABLE {acode}')
        for om in action.operable_masks:
            mask_str = ' '.join(om.mask)
            lines.append(
                f'{mask_str} _AGE >= {om.min_age} AND _AGE <= {om.max_age}'
            )
    path.write_text('\n'.join(lines) + '\n' if lines else '')
    return path


def emit_transitions(spec: ModelSpec, output_dir: Path) -> Path:
    """
    Emit the TRANSITIONS section file.

    Transitions are emitted in the order they appear in the spec (dict
    insertion order). Each transition case starts with ``*CASE {case}``,
    followed by ``*SOURCE`` and ``*TARGET`` entries.

    :param spec: The model specification.
    :param output_dir: Directory to write the file to.
    :return: Path to the written file.
    """
    path = output_dir / f'{spec.model_name}.trn'
    lines = []
    for case, trans in spec.transitions.items():
        lines.append(f'*CASE {case}')
        source_str = ' '.join(trans.source)
        lines.append(f'*SOURCE {source_str}')
        target_str = ' '.join(trans.target)
        proportion = int(trans.proportion * 100)
        lines.append(f'*TARGET {target_str} {proportion}')
    path.write_text('\n'.join(lines) + '\n' if lines else '')
    return path


def emit_outputs(spec: ModelSpec, output_dir: Path) -> Path:
    """
    Emit the OUTPUTS section file.

    Outputs are emitted in the order they appear in the spec. Each output
    declaration is on one line: ``*OUTPUT <code>(<theme>) <description>`` or
    ``*LEVEL <code>(<theme>) <description>``. The ``*SOURCE`` expression
    follows on the next line. Groups are emitted as ``*GROUP`` lines after
    all output declarations.

    Format matches what :py:meth:`ws3.forest.ForestModel._resolve_outputs_buffer`
    expects.

    :param spec: The model specification.
    :param output_dir: Directory to write the file to.
    :return: Path to the written file.
    """
    path = output_dir / f'{spec.model_name}.out'
    lines = []
    for output_spec in spec.outputs:
        keyword = '*LEVEL' if output_spec.is_level else '*OUTPUT'
        theme_part = f'({output_spec.theme_index})' if output_spec.theme_index else ''
        desc_part = f' {output_spec.description}' if output_spec.description else ''
        lines.append(f'{keyword} {output_spec.code}{theme_part}{desc_part}')
        if output_spec.expression:
            lines.append(f'*SOURCE {output_spec.expression}')
    for group in spec.output_groups:
        lines.append(f'*GROUP {group.name} {", ".join(group.output_codes)}')
    path.write_text('\n'.join(lines) + '\n' if lines else '')
    return path


def emit_all(spec: ModelSpec, output_dir: Path) -> dict[str, Path]:
    """
    Emit all supported section files.

    :param spec: The model specification.
    :param output_dir: Directory to write the files to.
    :return: Mapping of section name to path.
    """
    paths = {}
    if spec.themes or any(spec.areas.keys()):
        paths['landscape'] = emit_landscape(spec, output_dir)
    if spec.areas:
        paths['areas'] = emit_areas(spec, output_dir)
    if spec.yields:
        paths['yields'] = emit_yields(spec, output_dir)
    if spec.actions:
        paths['actions'] = emit_actions(spec, output_dir)
    if spec.transitions:
        paths['transitions'] = emit_transitions(spec, output_dir)
    if spec.outputs:
        paths['outputs'] = emit_outputs(spec, output_dir)
    return paths
