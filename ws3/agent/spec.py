"""
Construction-oriented typed model specification.

This module provides :py:class:`ModelSpec`, a typed, JSON-serialisable
specification for *building* a :py:class:`~ws3.forest.ForestModel` from
Woodstock-style section files. It is deliberately separate from
:py:class:`ws3.agent.themes.ModelContract`, which is an observation surface
that captures what an already-imported model looks like.

A :py:class:`ModelSpec` describes what to build:

- **themes**: the theme structure (names, basecodes, aggregates).
- **areas**: period-0 area inventory keyed by development-type key and age.
- **yields**: yield definitions with mask, type, and curve data.
- **actions**: action declarations with operable masks and expressions.
- **transitions**: transition cases with source, target, and theme replace.

The spec is construction-oriented: it carries enough information to emit
valid Woodstock section files and import them into a fresh model. It does
not carry runtime state (compiled curves, action execution traces, etc.).

Period handling is explicit: all age/period values are in *periods* as the
Woodstock format expects. The caller is responsible for converting to years
if needed. The builder applies the conversion when importing.

Transition features that this construction slice cannot emit are rejected
before emission. Action codes are currently accepted without type
classification; action metadata that is not emitted is reported in the
builder's loss record rather than guessed or silently discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

#: Action types ws3 supports.
SUPPORTED_ACTION_TYPES: frozenset[str] = frozenset({'harvest', 'thinning', 'regeneration'})

#: Transition features ws3 supports.
SUPPORTED_TRANSITION_FEATURES: frozenset[str] = frozenset({
    'theme_replace', 'theme_append', 'theme_mask',
})


class ModelSpecError(ValueError):
    """Base error for ModelSpec validation failures."""


class UnsupportedActionError(ModelSpecError):
    """An action type or feature in the spec is not supported by ws3."""


class UnsupportedTransitionError(ModelSpecError):
    """A transition feature in the spec is not supported by ws3."""


@dataclass(frozen=True)
class ThemeSpec:
    """
    One theme in a model specification.

    :param name: Theme name (e.g. ``'theme0'`` or a user-defined name).
    :param description: Human-readable description of the theme position.
    :param basecodes: Tuple of base codes for this theme.
    :param aggregates: Mapping of aggregate code to its member codes.
    """

    name: str
    description: str = ''
    basecodes: tuple[str, ...] = ()
    aggregates: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def codes(self) -> tuple[str, ...]:
        """All codes valid in this theme position (basecodes + aggregates)."""
        return self.basecodes + tuple(self.aggregates.keys())

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'basecodes': self.basecodes,
            'aggregates': dict(self.aggregates),
        }


@dataclass(frozen=True)
class YieldSpec:
    """
    One yield definition in a model specification.

    :param mask: Development-type mask (tuple of theme codes, ``'?'`` for wildcard).
    :param ytype: Yield type: ``'a'`` (age-based), ``'t'`` (time-based), or
        ``'complex'``.
    :param ynames: Tuple of yield component names.
    :param points: Mapping of yield component name to its curve points.
        For ``'a'`` and ``'t'`` types, values are lists of floats.
        For ``'complex'`` types, values are strings (expression formulas).
    """

    mask: tuple[str, ...]
    ytype: Literal['a', 't', 'complex']
    ynames: tuple[str, ...]
    points: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'mask': list(self.mask),
            'ytype': self.ytype,
            'ynames': list(self.ynames),
            'points': {k: list(v) if isinstance(v, (list, tuple)) else v
                       for k, v in self.points.items()},
        }


@dataclass(frozen=True)
class OperableMask:
    """
    One operable mask entry for an action.

    :param mask: Development-type mask (tuple of theme codes, ``'?'`` for wildcard).
    :param min_age: Minimum age (in periods) for this operable mask.
    :param max_age: Maximum age (in periods) for this operable mask.
    """

    mask: tuple[str, ...]
    min_age: int
    max_age: int

    def to_dict(self) -> dict[str, Any]:
        return {
            'mask': list(self.mask),
            'min_age': self.min_age,
            'max_age': self.max_age,
        }


@dataclass(frozen=True)
class OutputSpec:
    """
    One output declaration in a model specification.

    :param code: Output code (e.g. ``'totvol'``).
    :param theme_index: Optional theme index (e.g. ``'1'`` for first theme).
    :param description: Human-readable description.
    :param expression: Source expression string (opaque, not parsed).
    :param is_level: Whether this is a level output (vs. a computed output).
    """

    code: str
    theme_index: str | None = None
    description: str = ''
    expression: str = ''
    is_level: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            'code': self.code,
            'theme_index': self.theme_index,
            'description': self.description,
            'expression': self.expression,
            'is_level': self.is_level,
        }


@dataclass(frozen=True)
class OutputGroupSpec:
    """
    One output group in a model specification.

    :param name: Group name.
    :param output_codes: Tuple of output codes in this group.
    """

    name: str
    output_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'output_codes': list(self.output_codes),
        }


@dataclass(frozen=True)
class ActionSpec:
    """
    One action declaration in a model specification.

    :param acode: Action code (lowercase string).
    :param target_age: Target age for the action, or ``None`` for age-independent.
    :param description: Human-readable description.
    :param lock_exempt: Whether the action is exempt from lock constraints.
    :param operable_masks: Tuple of :py:class:`OperableMask` entries.
    """

    acode: str
    target_age: int | None = None
    description: str = ''
    lock_exempt: bool = False
    operable_masks: tuple[OperableMask, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            'acode': self.acode,
            'target_age': self.target_age,
            'description': self.description,
            'lock_exempt': self.lock_exempt,
            'operable_masks': [om.to_dict() for om in self.operable_masks],
        }


@dataclass(frozen=True)
class TransitionSpec:
    """
    One transition case in a model specification.

    :param case: Case identifier.
    :param source: Source development-type mask.
    :param target: Target development-type mask.
    :param action: Action code that triggers this transition.
    :param theme_replace: Theme replace expression (e.g. ``'_TH1'``).
    :param theme_append: Theme append expression.
    :param theme_mask: Theme mask to apply.
    :param proportion: Transition probability (0.0 to 1.0, defaults to 1.0).
    """

    case: str
    source: tuple[str, ...]
    target: tuple[str, ...]
    action: str
    theme_replace: str | None = None
    theme_append: str | None = None
    theme_mask: tuple[str, ...] | None = None
    proportion: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            'case': self.case,
            'source': list(self.source),
            'target': list(self.target),
            'action': self.action,
            'theme_replace': self.theme_replace,
            'theme_append': self.theme_append,
            'theme_mask': list(self.theme_mask) if self.theme_mask else None,
            'proportion': self.proportion,
        }


@dataclass
class ModelSpec:
    """
    A construction-oriented typed specification for a ForestModel.

    This is the inverse of :py:class:`ws3.agent.themes.ModelContract`: where
    the contract captures what *is*, the spec describes what to *build*.

    :param model_name: Base name for the model (used as file stem).
    :param base_year: Simulation base year.
    :param horizon: Number of periods in the simulation.
    :param period_length: Length of each period in years.
    :param max_age: Maximum stand age.
    :param area_epsilon: Minimum area to record.
    :param curve_epsilon: Minimum curve value to record.
    :param themes: List of :py:class:`ThemeSpec` instances.
    :param areas: Mapping of development-type key (tuple) to mapping of
        age (int) to area (float).
    :param yields: List of :py:class:`YieldSpec` instances.
    :param actions: Mapping of action code to :py:class:`ActionSpec`.
    :param transitions: Mapping of case identifier to :py:class:`TransitionSpec`.
    :param metadata: Arbitrary additional metadata (stored as-is).
    """

    model_name: str
    base_year: int
    horizon: int
    period_length: int
    max_age: int
    area_epsilon: float = 0.01
    curve_epsilon: float = 1e-06
    themes: tuple[ThemeSpec, ...] = ()
    areas: dict[tuple[str, ...], dict[int, float]] = field(default_factory=dict)
    yields: tuple[YieldSpec, ...] = ()
    actions: dict[str, ActionSpec] = field(default_factory=dict)
    transitions: dict[str, TransitionSpec] = field(default_factory=dict)
    outputs: tuple[OutputSpec, ...] = ()
    output_groups: tuple[OutputGroupSpec, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the spec after construction."""
        self._validate()

    def _validate(self) -> None:
        """Run structural validation on the spec."""
        # Validate theme names are unique.
        names = [t.name for t in self.themes]
        if len(names) != len(set(names)):
            raise ModelSpecError(f'Duplicate theme names: {names}')

        # Validate areas keys match theme count.
        n_themes = len(self.themes)
        for key in self.areas:
            if len(key) != n_themes:
                raise ModelSpecError(
                    f'Area key {key} has {len(key)} elements, expected {n_themes}'
                )

        # Validate yield masks match theme count.
        for y in self.yields:
            if len(y.mask) != n_themes:
                raise ModelSpecError(
                    f'Yield mask {y.mask} has {len(y.mask)} elements, expected {n_themes}'
                )

        # Validate transition sources/targets match theme count.
        for case, trans in self.transitions.items():
            if len(trans.source) != n_themes:
                raise ModelSpecError(
                    f'Transition {case} source has {len(trans.source)} elements, '
                    f'expected {n_themes}'
                )
            if len(trans.target) != n_themes:
                raise ModelSpecError(
                    f'Transition {case} target has {len(trans.target)} elements, '
                    f'expected {n_themes}'
                )

        # Validate action operable masks match theme count.
        for acode, action in self.actions.items():
            for om in action.operable_masks:
                if len(om.mask) != n_themes:
                    raise ModelSpecError(
                        f'Action {acode} operable mask {om.mask} has '
                        f'{len(om.mask)} elements, expected {n_themes}'
                    )
                if om.min_age < 0:
                    raise ModelSpecError(
                        f'Action {acode} operable mask has negative min_age: '
                        f'{om.min_age}'
                    )
                if om.max_age < om.min_age:
                    raise ModelSpecError(
                        f'Action {acode} operable mask has max_age < min_age: '
                        f'{om.max_age} < {om.min_age}'
                    )

        # Validate output codes are unique.
        output_codes = [o.code for o in self.outputs]
        if len(output_codes) != len(set(output_codes)):
            raise ModelSpecError(f'Duplicate output codes: {output_codes}')

        # Validate output group references exist.
        for group in self.output_groups:
            for code in group.output_codes:
                if code not in output_codes:
                    raise ModelSpecError(
                        f'Output group {group.name!r} references unknown '
                        f'output code {code!r}'
                    )

        # Validate transition features are supported.
        for case, trans in self.transitions.items():
            if trans.theme_replace and not trans.theme_replace.startswith('_TH'):
                raise UnsupportedTransitionError(
                    f'Transition {case} has unsupported theme_replace: '
                    f'{trans.theme_replace!r}'
                )
            if trans.theme_append:
                raise UnsupportedTransitionError(
                    f'Transition {case} has unsupported theme_append: '
                    f'{trans.theme_append!r}'
                )
            if trans.proportion < 0.0 or trans.proportion > 1.0:
                raise ModelSpecError(
                    f'Transition {case} has proportion out of range [0, 1]: '
                    f'{trans.proportion}'
                )

        # Validate yield point data shapes.
        for y in self.yields:
            if y.ytype in ('a', 't'):
                self._validate_yield_points(y)

    def _validate_yield_points(self, y: YieldSpec) -> None:
        """
        Validate that yield curve points are well-formed.

        Accepts two shapes:
          - Bare numeric sequences: ``[v0, v1, ...]``
          - Explicit ``(age, value)`` pair sequences: ``[(a0, v0), ...]``

        Mixed formats within a single yield are rejected. Pair ages must be
        strictly increasing and identical across all components.
        """
        if not y.points:
            return

        formats: set[str] = set()
        for yname, pts in y.points.items():
            if not pts:
                formats.add('empty')
                continue
            first = pts[0]
            if isinstance(first, (int, float)):
                formats.add('bare')
            elif isinstance(first, (list, tuple)) and len(first) == 2:
                if isinstance(first[0], (int, float)):
                    formats.add('pairs')
                else:
                    raise ModelSpecError(
                        f'Yield {y.mask}: component {yname!r} has malformed '
                        f'point data — first element {first!r} is not numeric'
                    )
            else:
                raise ModelSpecError(
                    f'Yield {y.mask}: component {yname!r} has unsupported '
                    f'point format: {type(first).__name__}'
                )

        non_empty = formats - {'empty'}
        if len(non_empty) > 1:
            raise ModelSpecError(
                f'Yield {y.mask}: mixed point formats detected '
                f'({dict(zip(y.ynames, [next(iter(non_empty)) for _ in non_empty]), strict=False)})'
            )

        if 'pairs' in non_empty:
            # Validate pair structure and age alignment.
            component_ages: dict[str, list[int]] = {}
            for yname, pts in y.points.items():
                ages: list[int] = []
                for i, item in enumerate(pts):
                    if not isinstance(item, (list, tuple)) or len(item) != 2:
                        raise ModelSpecError(
                            f'Yield {y.mask}: component {yname!r} pair at '
                            f'index {i} is not a (age, value) tuple'
                        )
                    age_val, _val = item
                    if not isinstance(age_val, (int, float)):
                        raise ModelSpecError(
                            f'Yield {y.mask}: component {yname!r} has '
                            f'non-numeric age at index {i}: {age_val!r}'
                        )
                    ages.append(int(age_val))
                if ages != sorted(ages) or len(ages) != len(set(ages)):
                    raise ModelSpecError(
                        f'Yield {y.mask}: component {yname!r} ages are not '
                        f'strictly increasing: {ages}'
                    )
                component_ages[yname] = ages

            # All components must share identical ages.
            ref_ages = component_ages[next(iter(component_ages))]
            for yname, ages in component_ages.items():
                if ages != ref_ages:
                    raise ModelSpecError(
                        f'Yield {y.mask}: component {yname!r} has ages '
                        f'{ages}, expected {ref_ages}'
                    )

    @staticmethod
    def _tuple_to_json_key(t: tuple[str, ...]) -> str:
        """Convert a development-type tuple key to a JSON-safe string."""
        return ' '.join(t)

    @staticmethod
    def _json_key_to_tuple(s: str) -> tuple[str, ...]:
        """Convert a JSON-safe string back to a development-type tuple key."""
        return tuple(s.split())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the spec to a JSON-compatible dict."""
        return {
            'model_name': self.model_name,
            'base_year': self.base_year,
            'horizon': self.horizon,
            'period_length': self.period_length,
            'max_age': self.max_age,
            'area_epsilon': self.area_epsilon,
            'curve_epsilon': self.curve_epsilon,
            'themes': [t.to_dict() for t in self.themes],
            'areas': {
                self._tuple_to_json_key(k): v for k, v in self.areas.items()
            },
            'yields': [y.to_dict() for y in self.yields],
            'actions': {
                k: v.to_dict()
                for k, v in self.actions.items()
            },
            'transitions': {k: v.to_dict() for k, v in self.transitions.items()},
            'outputs': [o.to_dict() for o in self.outputs],
            'output_groups': [g.to_dict() for g in self.output_groups],
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelSpec:
        """Deserialize a spec from a JSON-compatible dict."""
        themes = tuple(
            ThemeSpec(
                name=t['name'],
                description=t.get('description', ''),
                basecodes=tuple(t['basecodes']) if not isinstance(t['basecodes'], tuple) else t['basecodes'],
                aggregates={k: tuple(v) if not isinstance(v, tuple) else v
                          for k, v in t.get('aggregates', {}).items()},
            )
            for t in data.get('themes', [])
        )
        areas = {
            cls._json_key_to_tuple(k): {int(age): val for age, val in v.items()}
            for k, v in data.get('areas', {}).items()
        }
        yields = tuple(
            YieldSpec(
                mask=tuple(m['mask']),
                ytype=m['ytype'],
                ynames=tuple(m['ynames']),
                points=m['points'],
            )
            for m in data.get('yields', [])
        )
        actions = {
            k: ActionSpec(
                acode=v['acode'],
                target_age=v.get('target_age'),
                description=v.get('description', ''),
                lock_exempt=v.get('lock_exempt', False),
                operable_masks=tuple(
                    OperableMask(
                        mask=tuple(om['mask']),
                        min_age=om['min_age'],
                        max_age=om['max_age'],
                    )
                    for om in v.get('operable_masks', [])
                ),
            )
            for k, v in data.get('actions', {}).items()
        }
        transitions = {
            k: TransitionSpec(
                case=v['case'],
                source=tuple(v['source']),
                target=tuple(v['target']),
                action=v['action'],
                theme_replace=v.get('theme_replace'),
                theme_append=v.get('theme_append'),
                theme_mask=tuple(v['theme_mask']) if v.get('theme_mask') else None,
                proportion=v.get('proportion', 1.0),
            )
            for k, v in data.get('transitions', {}).items()
        }
        outputs = tuple(
            OutputSpec(
                code=v['code'],
                theme_index=v.get('theme_index'),
                description=v.get('description', ''),
                expression=v.get('expression', ''),
                is_level=v.get('is_level', False),
            )
            for v in data.get('outputs', [])
        )
        output_groups = tuple(
            OutputGroupSpec(
                name=v['name'],
                output_codes=tuple(v.get('output_codes', [])),
            )
            for v in data.get('output_groups', [])
        )
        return cls(
            model_name=data['model_name'],
            base_year=data['base_year'],
            horizon=data['horizon'],
            period_length=data['period_length'],
            max_age=data['max_age'],
            area_epsilon=data.get('area_epsilon', 0.01),
            curve_epsilon=data.get('curve_epsilon', 1e-06),
            themes=themes,
            areas=areas,
            yields=yields,
            actions=actions,
            transitions=transitions,
            outputs=outputs,
            output_groups=output_groups,
            metadata=data.get('metadata', {}),
        )

    def n_themes(self) -> int:
        """Number of themes in this spec."""
        return len(self.themes)

    def theme_codes(self, theme_index: int) -> tuple[str, ...]:
        """All codes valid in the given theme position."""
        return self.themes[theme_index].codes()

    def development_type_keys(self) -> list[tuple[str, ...]]:
        """All development-type keys that have area inventory."""
        return sorted(self.areas.keys())

    def has_area(self, key: tuple[str, ...], age: int) -> bool:
        """Whether the given development type and age has area inventory."""
        return key in self.areas and age in self.areas[key]

    def get_area(self, key: tuple[str, ...], age: int) -> float:
        """Get area for a development type and age, or 0.0."""
        return self.areas.get(key, {}).get(age, 0.0)
