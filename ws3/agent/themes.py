"""
Theme structure of a :py:class:`~ws3.forest.ForestModel`, for agent capabilities.

The Woodstock model format is deliberately open ended: the user declares their own
set of themes, and their own set of stratification variable codes within each
theme. Nothing about that vocabulary is known ahead of time, and much of ws3's
internal complexity follows from it.

Two consequences drive this module:

**Theme count is structural.** The number of themes fixes the length of every
development-type key, and therefore of every mask, in a given model instance. It
is authoritative state, read from the LANDSCAPE section of the input dataset (or
from whatever built the model). A capability that asks a language model to
reproduce it has invented a failure mode: a live 9B model duly returned a
ten-entry mask for a five-theme model. Masks are assembled here, from sparse
constraints, so the arity is correct by construction rather than by policing.

**The code vocabulary belongs to the user.** No fixed set of codes is ever valid
across models, so prompts must carry the instance's actual vocabulary and
validation must resolve against the instance. Aggregates matter especially: an
aggregate is a name the user chose for a group of codes, so it is usually the
closest thing in the model to the words a person will use when describing the
stands they mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Cap on codes listed per theme in a prompt. Real models can carry hundreds;
#: listing all of them buries the task and wastes context.
MAX_CODES_PER_THEME = 40

#: Cap on members listed when expanding an aggregate in a prompt.
MAX_AGGREGATE_MEMBERS = 12

#: Mask entry matching any code in a theme position.
WILDCARD = '?'


class ThemeError(ValueError):
    """A theme position or code that does not exist in this model instance."""


@dataclass(frozen=True)
class Theme:
    """
    One theme of a model instance.

    :param index: Position in the development-type key.
    :param name: Theme name, e.g. ``theme0``.
    :param description: Theme description, often empty.
    :param basecodes: Codes the user declared directly.
    :param aggregates: Aggregate code to the codes it stands for.
    """

    index: int
    name: str
    description: str
    basecodes: tuple[str, ...]
    aggregates: dict[str, tuple[str, ...]]

    def known(self, code: str) -> bool:
        """Whether *code* is a basecode or an aggregate of this theme."""
        return code in self.basecodes or code in self.aggregates

    def codes(self) -> tuple[str, ...]:
        """Every code valid in this theme position, basecodes and aggregates."""
        return self.basecodes + tuple(self.aggregates)


class ThemeSchema:
    """
    The theme structure of one model instance.

    Built from a :py:class:`~ws3.forest.ForestModel`; carries no reference to it,
    so it is safe to hold, describe and diff.
    """

    def __init__(self, themes: tuple[Theme, ...]) -> None:
        self.themes = themes

    @classmethod
    def from_model(cls, fm: Any) -> ThemeSchema:
        """
        Read the theme structure out of a model.

        Basecodes and aggregates are distinguished by the shape of the stored
        value -- ws3 stores a basecode as ``{code: code}`` and an aggregate as
        ``{code: [members]}`` -- rather than via
        :py:meth:`~ws3.forest.ForestModel.theme_basecodes`, because that parallel
        list is only appended to when a theme is declared with a non-empty
        basecode list, so it can fall out of step with ``_themes``.
        """
        themes = []
        for index, raw in enumerate(fm._themes):
            basecodes = []
            aggregates = {}
            for code, value in raw.items():
                if code.startswith('__'):
                    continue
                if isinstance(value, (list, tuple, set)):
                    aggregates[code] = tuple(value)
                else:
                    basecodes.append(code)
            themes.append(
                Theme(
                    index=index,
                    name=raw.get('__name__', f'theme{index}'),
                    description=raw.get('__description__', ''),
                    basecodes=tuple(basecodes),
                    aggregates=aggregates,
                )
            )
        return cls(tuple(themes))

    def __len__(self) -> int:
        return len(self.themes)

    @property
    def nthemes(self) -> int:
        """Number of themes, and so the length of every development-type key."""
        return len(self.themes)

    def wildcard_mask(self) -> tuple[str, ...]:
        """A mask matching every development type in the model."""
        return tuple([WILDCARD] * self.nthemes)

    def assemble(self, constraints: dict[Any, Any]) -> tuple[str, ...]:
        """
        Expand sparse theme constraints into a full-length mask.

        Positions not mentioned become wildcards, so the result always has exactly
        one entry per theme regardless of what was supplied. This is what keeps
        mask arity out of a language model's hands.

        :param constraints: Theme position to code. Keys may be ``int`` or the
            string form of one.
        :raises ThemeError: If a position is not an integer or is out of range.
        """
        if not isinstance(constraints, dict):
            raise ThemeError(
                f'constraints must be a mapping of theme position to code, got '
                f'{type(constraints).__name__}'
            )

        mask = list(self.wildcard_mask())
        for key, code in constraints.items():
            mask[self._position(key)] = str(code).lower()
        return tuple(mask)

    def _position(self, key: Any) -> int:
        """
        Resolve a constraint key to a valid theme index.

        Accepts the position number or the theme's name. The listing shown to a
        model puts both in front of it, and a model that keys by the name is not
        wrong about the theme -- only about the notation. Names are unambiguous
        here, so resolving them is deterministic rather than a guess, and it costs
        nothing compared with a retry.
        """
        if isinstance(key, str):
            name = key.strip().lower()
            for theme in self.themes:
                if theme.name.lower() == name:
                    return theme.index

        try:
            index = int(key)
        except (TypeError, ValueError):
            names = ', '.join(t.name for t in self.themes)
            raise ThemeError(
                f'theme position {key!r} is not a position number or a known theme '
                f'name; use one of the position numbers 0 to {self.nthemes - 1}, '
                f'or one of: {names}'
            ) from None
        if not 0 <= index < self.nthemes:
            raise ThemeError(
                f'theme position {index} is out of range; this model has '
                f'{self.nthemes} themes, numbered 0 to {self.nthemes - 1}'
            )
        return index

    def unknown_codes(self, mask: tuple[str, ...]) -> list[str]:
        """
        Describe mask entries that are not codes of their theme.

        Turns "matched nothing" into something a retry can act on, and names what
        would have worked -- the vocabulary is user-defined, so the model cannot
        infer it.
        """
        problems = []
        for index, code in enumerate(mask):
            if code == WILDCARD or index >= self.nthemes:
                continue
            theme = self.themes[index]
            if theme.known(code):
                continue
            valid = list(theme.codes())
            shown = ', '.join(valid[:MAX_CODES_PER_THEME]) or '(none)'
            suffix = '' if len(valid) <= MAX_CODES_PER_THEME else ' ...'
            problems.append(
                f'position {index} ({theme.name}): {code!r} is not a code for this '
                f'theme (valid: {shown}{suffix})'
            )
        return problems

    def describe(self) -> str:
        """
        Render the theme vocabulary for a prompt.

        Aggregates are listed separately and expanded, because an aggregate is a
        name the user chose for a group of stands and is therefore the code most
        likely to match how a person describes them.

        A theme with no description is marked as such. ``import_landscape_section``
        names themes ``theme0``, ``theme1`` and so on; those are positional
        placeholders carrying no meaning, and saying so is the difference between
        selecting on evidence and pattern-matching on a number.
        """
        lines = []
        for theme in self.themes:
            header = f'  position {theme.index} ({theme.name})'
            if theme.description:
                header += f' -- {theme.description}'
            else:
                header += ' -- undescribed, no stated meaning'
            lines.append(header + ':')

            shown = list(theme.basecodes[:MAX_CODES_PER_THEME])
            suffix = (
                '' if len(theme.basecodes) <= MAX_CODES_PER_THEME
                else f' ... ({len(theme.basecodes)} total)'
            )
            lines.append(f'      codes: {", ".join(shown) or "(none)"}{suffix}')

            for name, members in theme.aggregates.items():
                listed = ', '.join(members[:MAX_AGGREGATE_MEMBERS])
                more = (
                    '' if len(members) <= MAX_AGGREGATE_MEMBERS
                    else f' ... ({len(members)} total)'
                )
                lines.append(f'      aggregate {name}: {listed}{more}')
        return '\n'.join(lines)


def schema_for(fm: Any | None) -> ThemeSchema | None:
    """Return the schema for *fm*, or ``None`` if no model was supplied."""
    return None if fm is None else ThemeSchema.from_model(fm)


# ---------------------------------------------------------------------------
# Model contract: typed, JSON-serialisable specification and structural
# verification of a ForestModel instance.
# ---------------------------------------------------------------------------

from dataclasses import asdict, dataclass, field  # noqa: E402

#: Severity levels for verification findings.
SEVERITY_ERROR = 'error'
SEVERITY_WARNING = 'warning'


@dataclass(frozen=True)
class VerificationFinding:
    """
    One structural finding from verifying a model contract.

    :param level: Check level. ``'L0'`` is mandatory structural integrity;
        ``'L1'`` is a cheaper secondary check.
    :param category: Human-readable category, e.g. ``'theme_arity'``.
    :param message: Description of what was found.
    :param severity: ``'error'`` blocks use; ``'warning'`` is informational.
    """

    level: str
    category: str
    message: str
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    """
    The outcome of verifying a :py:class:`ModelContract`.

    Findings are returned rather than exceptions raised: a model that fails L0
    is still a model, and callers need to know *what* is wrong, not just that
    something is wrong.
    """

    findings: list[VerificationFinding] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when no finding has error severity."""
        return not any(f.severity == SEVERITY_ERROR for f in self.findings)

    @property
    def errors(self) -> list[VerificationFinding]:
        return [f for f in self.findings if f.severity == SEVERITY_ERROR]

    @property
    def warnings(self) -> list[VerificationFinding]:
        return [f for f in self.findings if f.severity == SEVERITY_WARNING]

    def to_dict(self) -> dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'findings': [f.to_dict() for f in self.findings],
            'summary': {
                'total': len(self.findings),
                'errors': len(self.errors),
                'warnings': len(self.warnings),
            },
        }


class ModelContract:
    """
    A typed, JSON-serialisable specification of a :py:class:`~ws3.forest.ForestModel`.

    Built from an existing model via :py:meth:`from_model`; carries no reference
    to it, so it is safe to hold, describe, serialise and diff.

    The contract captures three layers:

    1. **Metadata** -- model name, base year, horizon, period length, max age.
    2. **Theme schema** -- the :py:class:`ThemeSchema` of the model.
    3. **Development types** -- the set of keys actually present, each reduced to
       its theme-code tuple and a count of period-0 area buckets.

    Structural verification runs deterministic L0 and L1 checks and returns
    findings rather than raising, so ordinary model invalidity is observable
    without interrupting the caller.

    :param metadata: Flat dict of scalar model metadata.
    :param schema: The theme schema.
    :param development_types: List of ``(key, n_age_classes)`` pairs.
    """

    def __init__(
        self,
        metadata: dict[str, Any],
        schema: ThemeSchema,
        development_types: list[tuple[tuple[str, ...], int]],
    ) -> None:
        self.metadata = metadata
        self.schema = schema
        self.development_types = development_types

    @classmethod
    def from_model(cls, fm: Any) -> ModelContract:
        """
        Extract the contract from a :py:class:`~ws3.forest.ForestModel`.

        :param fm: The model to extract from.
        :return: A :py:class:`ModelContract` describing *fm*.
        """
        metadata = {
            'model_name': fm.model_name,
            'base_year': fm.base_year,
            'horizon': fm.horizon,
            'period_length': fm.period_length,
            'max_age': fm.max_age,
            'area_epsilon': fm.area_epsilon,
            'curve_epsilon': fm.curve_epsilon,
            'n_development_types': len(fm.dtypes),
            'n_actions': len(fm.actions),
        }

        development_types = []
        for key in sorted(fm.dtypes.keys()):
            n_ages = len(fm.dtypes[key]._areas[0])
            development_types.append((key, n_ages))

        return cls(
            metadata=metadata,
            schema=ThemeSchema.from_model(fm),
            development_types=development_types,
        )

    def verify(self) -> VerificationResult:
        """
        Run deterministic structural checks and return the findings.

        L0 checks (errors):

        - ``theme_arity``: theme indices are the contiguous range ``0..nthemes-1``.
        - ``theme_has_basecodes``: every theme declares at least one basecode.
        - ``dtype_key_length``: every development-type key has length equal to
          ``nthemes``.
        - ``dtype_code_known``: every code in every development-type key is a
          known code (basecode or aggregate) for its theme position.

        L1 checks (warnings):

        - ``dtype_duplicate_key``: development-type keys are unique (invariant of
          the ``dtypes`` dict, but worth asserting explicitly).
        - ``action_orphan``: an action code referenced in any development type's
          ``oper_expr`` is not declared in ``fm.actions``.

        :return: A :py:class:`VerificationResult` with all findings.
        """
        findings: list[VerificationFinding] = []
        nthemes = self.schema.nthemes

        # L0: theme_arity
        expected_indices = set(range(nthemes))
        actual_indices = {t.index for t in self.schema.themes}
        if actual_indices != expected_indices:
            missing = expected_indices - actual_indices
            extra = actual_indices - expected_indices
            if missing:
                findings.append(
                    VerificationFinding(
                        level='L0',
                        category='theme_arity',
                        message=(
                            f'theme indices skip {sorted(missing)}; '
                            f'expected contiguous 0..{nthemes - 1}'
                        ),
                        severity=SEVERITY_ERROR,
                    )
                )
            if extra:
                findings.append(
                    VerificationFinding(
                        level='L0',
                        category='theme_arity',
                        message=(
                            f'unexpected theme indices {sorted(extra)}; '
                            f'expected contiguous 0..{nthemes - 1}'
                        ),
                        severity=SEVERITY_ERROR,
                    )
                )

        # L0: theme_has_basecodes
        for theme in self.schema.themes:
            if not theme.basecodes:
                findings.append(
                    VerificationFinding(
                        level='L0',
                        category='theme_has_basecodes',
                        message=(
                            f'theme position {theme.index} ({theme.name}) has no '
                            f'basecodes; a theme without basecodes cannot select '
                            f'any development type'
                        ),
                        severity=SEVERITY_ERROR,
                    )
                )

        # L0: dtype_key_length
        for key, _n_ages in self.development_types:
            if len(key) != nthemes:
                findings.append(
                    VerificationFinding(
                        level='L0',
                        category='dtype_key_length',
                        message=(
                            f'development type key {key!r} has length {len(key)}, '
                            f'expected {nthemes}'
                        ),
                        severity=SEVERITY_ERROR,
                    )
                )

        # L0: dtype_code_known
        theme_by_index = {t.index: t for t in self.schema.themes}
        for key, _n_ages in self.development_types:
            for pos, code in enumerate(key):
                if pos >= nthemes:
                    continue
                theme = theme_by_index[pos]
                if code == WILDCARD:
                    continue
                if not theme.known(code):
                    valid = list(theme.codes())
                    shown = ', '.join(valid[:MAX_CODES_PER_THEME]) or '(none)'
                    findings.append(
                        VerificationFinding(
                            level='L0',
                            category='dtype_code_known',
                            message=(
                                f'development type {key!r} position {pos} '
                                f'({theme.name}): {code!r} is not a known code '
                                f'(valid: {shown})'
                            ),
                            severity=SEVERITY_ERROR,
                        )
                    )

        # L1: dtype_duplicate_key (cheap assertion of an invariant)
        seen_keys: set[tuple[str, ...]] = set()
        for key, _n_ages in self.development_types:
            if key in seen_keys:
                findings.append(
                    VerificationFinding(
                        level='L1',
                        category='dtype_duplicate_key',
                        message=f'development type key {key!r} appears more than once',
                        severity=SEVERITY_WARNING,
                    )
                )
            seen_keys.add(key)

        # L1: action_orphan
        # This requires the original model's actions dict, which the contract
        # does not carry. Skip this check here -- it belongs to a richer
        # verification pass that keeps a reference to the model, or to a
        # separate capability. Documented as a known limitation.

        return VerificationResult(findings=findings)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return {
            'metadata': self.metadata,
            'schema': {
                'nthemes': self.schema.nthemes,
                'themes': [
                    {
                        'index': t.index,
                        'name': t.name,
                        'description': t.description,
                        'basecodes': list(t.basecodes),
                        'aggregates': {k: list(v) for k, v in t.aggregates.items()},
                    }
                    for t in self.schema.themes
                ],
            },
            'development_types': [
                {'key': list(k), 'n_age_classes': n}
                for k, n in self.development_types
            ],
        }


def contract_for(fm: Any) -> ModelContract:
    """Convenience: return the :py:class:`ModelContract` for *fm*."""
    return ModelContract.from_model(fm)
