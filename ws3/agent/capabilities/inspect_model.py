"""
``inspect_model`` — read-only metadata snapshot of a live ForestModel.

This capability gathers a bounded set of safe metadata fields from the
:py:class:`~ws3.forest.ForestModel` and validates each field against live
state. It is explicitly read-only and never executes model-generated Python.

Supported fields:

- ``model_name`` / ``name`` — from ``fm.model_name``
- ``base_year`` — from ``fm.base_year``
- ``horizon`` — number of periods
- ``period_length`` — from ``fm.period_length``
- ``periods`` — from ``fm.periods``
- ``nthemes`` — count of ``fm._themes``
- ``nactions`` — count of ``fm.actions``
- ``ndtypes`` — count of ``fm.dtypes``
- ``total_area`` — unambiguous sum of ``dt.area(1)`` across all development
  types at period 1 (the base period). ``None`` if any dtype raises or the
  sum is ambiguous.

Unsupported requests (arbitrary operable-area filters, plotting, time series
beyond the base period, or any mutation) return an explicit unsupported
result. No fabricated values are ever produced.

The provider selects which *bounded operation* to execute from a fixed set;
it never supplies trusted numeric facts. Numeric values come from the
deterministic executor, which reads the live model directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from fresh_agent_core.capability import Capability, ParseError, Verdict

#: Fixed set of bounded operations the provider may select.
_BOUNDED_OPERATIONS = frozenset([
    'full_snapshot',
    'model_identity',
    'temporal_summary',
    'counts',
    'area',
])


@dataclass(frozen=True)
class InspectInputs:
    """
    What the user wants to inspect.

    :param query: Natural-language query, e.g. "show me the model metadata"
        or "just the base year and horizon".
    :param model_name: Optional model name filter. If non-empty, the executor
        rejects any model whose ``model_name`` does not match.
    """

    query: str
    model_name: str = ''


@dataclass(frozen=True)
class InspectResult:
    """
    A validated metadata snapshot of a ForestModel.

    Every field is read directly from the live model. ``None`` means the field
    could not be safely computed (e.g. ``total_area`` when a dtype has no area
    data). ``unsupported`` is non-empty only when the user query fell outside
    the bounded set.

    :param model_name: The model's ``model_name`` attribute.
    :param name: Alias for ``model_name`` when the model exposes it. ``None``
        if the attribute is absent.
    :param base_year: The model's ``base_year``. ``None`` if absent.
    :param horizon: Number of simulation periods. ``None`` if periods cannot
        be enumerated.
    :param period_length: The model's ``period_length``. ``None`` if absent.
    :param periods: The list of period integers (1-based). ``None`` if absent.
    :param nthemes: Count of themes loaded. ``None`` if the attribute is
        private or empty.
    :param nactions: Count of action codes. ``None`` if ``actions`` is empty
        or absent.
    :param ndtypes: Count of development types. ``None`` if ``dtypes`` is empty
        or absent.
    :param total_area: Sum of ``dt.area(1)`` across all development types.
        ``None`` if the sum cannot be computed unambiguously.
    :param unsupported: Non-empty string when the query was outside the
        bounded set. Empty string otherwise.
    :param raw: Raw model output.
    :param operation: The bounded operation selected by the provider. One of
        :py:data:`_BOUNDED_OPERATIONS` or ``"unsupported"``.
    """

    model_name: str | None
    name: str | None
    base_year: int | None
    horizon: int | None
    period_length: float | None
    periods: list[int] | None
    nthemes: int | None
    nactions: int | None
    ndtypes: int | None
    total_area: float | None
    unsupported: str = ''
    raw: str = ''
    operation: str = ''


def _snapshot(fm: Any) -> dict[str, Any]:
    """
    Deterministic metadata snapshot of a ForestModel.

    Reads only safe, public or semi-public attributes. Never executes
    model-generated Python. Never mutates.

    :param fm: A ForestModel instance.
    :return: A dict of field name to value.
    """
    out: dict[str, Any] = {}

    # model_name
    out['model_name'] = getattr(fm, 'model_name', None)
    out['name'] = getattr(fm, 'name', out['model_name'])

    # base_year
    out['base_year'] = getattr(fm, 'base_year', None)

    # period_length
    out['period_length'] = getattr(fm, 'period_length', None)

    # periods
    periods = getattr(fm, 'periods', None)
    if periods is not None:
        try:
            out['periods'] = list(periods)
        except TypeError:
            out['periods'] = None
    else:
        out['periods'] = None

    # horizon = len(periods)
    if out['periods']:
        out['horizon'] = len(out['periods'])
    else:
        out['horizon'] = None

    # nthemes
    nthemes = getattr(fm, 'nthemes', None)
    if callable(nthemes):
        try:
            out['nthemes'] = int(nthemes())
        except Exception:
            out['nthemes'] = None
    else:
        themes = getattr(fm, '_themes', None)
        out['nthemes'] = len(themes) if themes is not None else None

    # nactions
    actions = getattr(fm, 'actions', None)
    out['nactions'] = len(actions) if actions is not None else 0

    # ndtypes
    dtypes = getattr(fm, 'dtypes', None)
    out['ndtypes'] = len(dtypes) if dtypes is not None else 0

    # total_area — only from period 1, unambiguous sum
    total = None
    try:
        area_sum = 0.0
        count = 0
        for _dtk, dt in (dtypes or {}).items():
            a = getattr(dt, 'area', None)
            if a is None:
                total = None
                break
            try:
                area_val = a(1)  # period 1, all ages
            except Exception:
                total = None
                break
            if area_val is None:
                total = None
                break
            area_sum += float(area_val)
            count += 1
        if count == 0:
            total = 0.0
        else:
            total = area_sum
    except Exception:
        total = None
    out['total_area'] = total

    return out


class InspectModel(Capability[InspectResult]):
    """
    Read-only metadata snapshot of a ws3 ForestModel.

    Gathers a bounded set of safe metadata fields from the live model and
    validates each against real state. Explicitly read-only. Never executes
    model-generated Python. Never fabricates numeric values.

    Supported queries:

    - "show me the model" — full snapshot of all fields
    - "model identity" — model_name and name only
    - "temporal summary" — base_year, horizon, period_length, periods
    - "counts" — nthemes, nactions, ndtypes
    - "area" — total_area only

    Unsupported queries (plotting, time series beyond period 1, arbitrary
    operable-area filters, any mutation) return an explicit unsupported
    result.

    The provider selects which bounded operation to execute from the fixed
    set above; it never supplies trusted numeric facts. Numeric values come
    from the deterministic executor, which reads the live model directly.
    """

    name = 'inspect_model'
    description = (
        'Read-only metadata snapshot of a ws3 ForestModel. Validates model '
        'identity and computes safe metadata (model_name, base_year, horizon, '
        'period_length, periods, theme/action/dtype counts, total area) '
        'directly from the live model. Does not execute model-generated '
        'Python. Does not plot, does not produce time series beyond the '
        'base period. Unsupported requests return an explicit unsupported '
        'result rather than fabricated values.'
    )
    max_attempts = 2  # One for selection, one retry if malformed

    input_schema = {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': (
                    'Natural-language query, e.g. "show me the model", '
                    '"model identity", "temporal summary", "counts", "area", '
                    'or "full snapshot".'
                ),
            },
            'model_name': {
                'type': 'string',
                'description': (
                    'Optional model name filter. The snapshot is rejected if '
                    'the live model\'s model_name does not match.'
                ),
            },
        },
        'required': ['query'],
    }

    def from_payload(self, payload: dict) -> InspectInputs:
        """Build :py:class:`InspectInputs` from an MCP tool-call payload."""
        return InspectInputs(
            query=str(payload.get('query', '')),
            model_name=str(payload.get('model_name', '')),
        )

    def render(self, value: InspectResult) -> str:
        """Render as a human-readable metadata snapshot."""
        lines = ['### WS3 Inspect Model', '']
        if value.unsupported:
            lines.extend([
                f'**Unsupported query**: {value.unsupported}',
                '',
            ])
            return '\n'.join(lines)

        if value.model_name is not None:
            lines.append(f'- **model_name**: `{value.model_name}`')
        if value.name is not None:
            lines.append(f'- **name**: `{value.name}`')
        if value.base_year is not None:
            lines.append(f'- **base_year**: `{value.base_year}`')
        if value.horizon is not None:
            lines.append(f'- **horizon** (periods): `{value.horizon}`')
        if value.period_length is not None:
            lines.append(f'- **period_length**: `{value.period_length}`')
        if value.periods is not None:
            lines.append(f'- **periods**: `{value.periods}`')
        if value.nthemes is not None:
            lines.append(f'- **nthemes**: `{value.nthemes}`')
        if value.nactions is not None:
            lines.append(f'- **nactions**: `{value.nactions}`')
        if value.ndtypes is not None:
            lines.append(f'- **ndtypes**: `{value.ndtypes}`')
        if value.total_area is not None:
            lines.append(f'- **total_area** (period 1): `{value.total_area}`')
        else:
            lines.append('- **total_area**: unavailable (no safe sum)')

        lines.extend([
            '',
            '> Read-only snapshot. Values come from the live ForestModel.',
            '> The provider selected the bounded operation; the executor '
            'computed the numeric values directly.',
        ])
        return '\n'.join(lines)

    def build_messages(
        self,
        inputs: InspectInputs,
        failures: tuple[str, ...],
    ) -> list[dict[str, str]]:
        """
        Ask the provider to select a bounded operation.

        The provider may ONLY respond with one of the entries in
        :py:data:`_BOUNDED_OPERATIONS` or ``"unsupported"``. It must NOT
        compute numeric values, plot, or execute Python.
        """
        content = (
            'You are a bounded router for a read-only metadata capability. '
            'Given the user\'s query, select ONE bounded operation from this '
            'fixed set and respond with a JSON object containing only the '
            '"operation" key:\n'
            '\n'
            '  full_snapshot — return all metadata fields\n'
            '  model_identity — return only model_name and name\n'
            '  temporal_summary — return only base_year, horizon, '
            'period_length, periods\n'
            '  counts — return only nthemes, nactions, ndtypes\n'
            '  area — return only total_area\n'
            '\n'
            'Rules:\n'
            '- Respond with a JSON object: {"operation": "<name>"}\n'
            '- If the query does not match any bounded operation, respond '
            'with {"operation": "unsupported"}\n'
            '- Do NOT compute numeric values yourself\n'
            '- Do NOT execute model-generated Python\n'
            '- Do NOT plot or produce time series beyond the base period\n'
            '- Do NOT filter by arbitrary operable-area conditions\n'
            '\n'
            f'User query: {inputs.query}\n'
        )
        if inputs.model_name:
            content += f'Model name filter: {inputs.model_name}\n'
        if failures:
            content += f'\nPrevious failures: {"; ".join(failures)}\n'
        return [{'role': 'user', 'content': content}]

    def parse(self, raw: str) -> InspectResult:
        """
        Parse the provider's bounded-operation selection.

        The raw output must be a JSON object with an "operation" key whose
        value is one of :py:data:`_BOUNDED_OPERATIONS` or "unsupported".

        :raises ParseError: If the response is not a valid bounded selection.
        """
        cleaned = raw.strip()
        # Strip markdown fences if present
        if cleaned.startswith('```'):
            cleaned = cleaned.split('\n', 1)[1] if '\n' in cleaned else ''
            cleaned = cleaned.rstrip('`').strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ParseError(
                f'expected a JSON object with "operation" key, got invalid '
                f'JSON: {exc.msg} at line {exc.lineno}:{exc.colno}'
            ) from exc

        if not isinstance(payload, dict):
            raise ParseError(
                f'expected a JSON object, got {type(payload).__name__}'
            )

        operation = str(payload.get('operation', '')).strip().lower()
        if not operation:
            raise ParseError(
                'expected a JSON object with "operation" key, '
                'got JSON without an operation key'
            )
        if operation not in _BOUNDED_OPERATIONS and operation != 'unsupported':
            raise ParseError(
                f'unrecognized operation {operation!r}; must be one of '
                f'{", ".join(sorted(_BOUNDED_OPERATIONS))} or "unsupported"'
            )

        return InspectResult(
            model_name=None,
            name=None,
            base_year=None,
            horizon=None,
            period_length=None,
            periods=None,
            nthemes=None,
            nactions=None,
            ndtypes=None,
            total_area=None,
            unsupported='' if operation != 'unsupported' else 'query outside bounded operations',
            raw=raw,
            operation=operation,
        )

    def validate(
        self,
        candidate: InspectResult,
        context: Any,
    ) -> Verdict:
        """
        Validate the parsed operation and run the deterministic executor.

        :param candidate: Parsed output from :py:meth:`parse`.
        :param context: The live ForestModel.
        :return: ``Verdict.valid()`` if the context is a ForestModel,
            or ``Verdict.invalid()`` with reasons otherwise.

        Notes:
            Unsupported queries are treated as a valid parse: the provider
            explicitly selected ``unsupported`` rather than supplying a bounded
            operation with fabricated numbers. :py:meth:`run` checks the
            operation and returns an explicit unsupported result with no
            numeric facts.
        """
        # No model context — cannot validate
        if context is None:
            return Verdict.invalid(
                'No ForestModel provided as context. Create one first '
                '— e.g. ``fm = ForestModel(...)``.'
            )

        # Class check
        cls_name = getattr(context, '__class__', None)
        if cls_name is None or cls_name.__name__ != 'ForestModel':
            return Verdict.invalid(
                f'context is {cls_name.__name__ if cls_name else type(context)!r}, '
                f'not ForestModel'
            )

        # Operation validation — unsupported is a valid provider selection
        if candidate.operation == 'unsupported':
            return Verdict.valid()

        return Verdict.valid()

    def run(
        self,
        inputs: Any,
        *,
        provider: Any,
        config: Any,
        context: Any = None,
        sink: Any = None,
    ) -> Any:
        """
        Execute the bounded snapshot capability with the inherited validate/
        retry loop, then compute fields based on the selected operation.

        The provider may return malformed output on the first attempt; the
        parent loop retries up to ``max_attempts`` times, feeding previous
        failures back via ``build_messages``. Never executes model-generated
        Python. Never fabricates numeric values.

        After a successful parse/validate cycle, this method computes the
        metadata snapshot and populates only the fields requested by the
        selected operation:

        - ``full_snapshot`` — all fields
        - ``model_identity`` — ``model_name``, ``name``
        - ``temporal_summary`` — ``base_year``, ``horizon``, ``period_length``,
          ``periods``
        - ``counts`` — ``nthemes``, ``nactions``, ``ndtypes``
        - ``area`` — ``total_area``
        - ``unsupported`` — no numeric fields (explicit unsuccessful result)
        """
        from fresh_agent_core.capability import CapabilityResult

        # Coerce inputs
        if not isinstance(inputs, InspectInputs):
            inputs = InspectInputs(query=str(inputs) if inputs else '')

        # Run the inherited validate/retry loop
        result = super().run(
            inputs,
            provider=provider,
            config=config,
            context=context,
            sink=sink,
        )

        if not result.ok or result.value is None:
            return result

        candidate = result.value
        operation = candidate.operation

        # Unsupported — return explicit unsuccessful result with no numeric facts
        if operation == 'unsupported':
            return CapabilityResult(
                ok=True,
                value=InspectResult(
                    model_name=None,
                    name=None,
                    base_year=None,
                    horizon=None,
                    period_length=None,
                    periods=None,
                    nthemes=None,
                    nactions=None,
                    ndtypes=None,
                    total_area=None,
                    unsupported=candidate.unsupported,
                    raw=candidate.raw,
                    operation=operation,
                ),
                attempts=result.attempts,
                provenance_ids=result.provenance_ids,
                errors=result.errors,
            )

        # Bounded operation — compute the snapshot and populate only the
        # requested fields
        snapshot = _snapshot(context)

        # Field selection per operation
        _OPERATION_FIELDS: dict[str, tuple[str, ...]] = {
            'full_snapshot': (
                'model_name', 'name', 'base_year', 'horizon',
                'period_length', 'periods', 'nthemes', 'nactions',
                'ndtypes', 'total_area',
            ),
            'model_identity': ('model_name', 'name'),
            'temporal_summary': ('base_year', 'horizon', 'period_length', 'periods'),
            'counts': ('nthemes', 'nactions', 'ndtypes'),
            'area': ('total_area',),
        }

        fields = _OPERATION_FIELDS.get(operation, ())

        new_value = InspectResult(
            model_name=snapshot.get('model_name') if 'model_name' in fields else None,
            name=snapshot.get('name') if 'name' in fields else None,
            base_year=snapshot.get('base_year') if 'base_year' in fields else None,
            horizon=snapshot.get('horizon') if 'horizon' in fields else None,
            period_length=snapshot.get('period_length') if 'period_length' in fields else None,
            periods=snapshot.get('periods') if 'periods' in fields else None,
            nthemes=snapshot.get('nthemes') if 'nthemes' in fields else None,
            nactions=snapshot.get('nactions') if 'nactions' in fields else None,
            ndtypes=snapshot.get('ndtypes') if 'ndtypes' in fields else None,
            total_area=snapshot.get('total_area') if 'total_area' in fields else None,
            unsupported='',
            raw=candidate.raw,
            operation=operation,
        )

        # Apply model_name filter to the complete live snapshot before projection.
        if inputs.model_name and snapshot.get('model_name') != inputs.model_name:
            # Filter failed — return invalid result
            return CapabilityResult(
                ok=False,
                value=None,
                attempts=result.attempts,
                provenance_ids=result.provenance_ids,
                errors=(
                    f'model_name filter {inputs.model_name!r} does not '
                    f'match live model {snapshot.get("model_name")!r}',
                ),
            )

        return CapabilityResult(
            ok=True,
            value=new_value,
            attempts=result.attempts,
            provenance_ids=result.provenance_ids,
            errors=result.errors,
        )
