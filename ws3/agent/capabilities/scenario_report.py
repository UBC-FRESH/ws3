"""Deterministic inventory and products reporting for a bundled WS3 scenario."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fresh_agent_core.capability import Capability, CapabilityResult, ParseError, Verdict

__all__ = [
    'ScenarioModelIdentity',
    'ScenarioReport',
    'ScenarioReportInputs',
    'ScenarioReportResult',
    'ScenarioReportRow',
    'ScheduleProvenance',
    'report_scenario_inventory_products',
]


@dataclass(frozen=True)
class ScenarioReportInputs:
    """Explicit model and bundled schedule selection for one report."""

    model_path: str
    model_name: str
    schedule_path: str | None = None


@dataclass(frozen=True)
class ScenarioModelIdentity:
    """Identity and loaded structure of the model used for the report."""

    model_name: str
    model_path: str
    base_year: int | None
    horizon: int | None
    period_length: float | None
    nthemes: int | None
    nactions: int | None
    ndtypes: int | None


@dataclass(frozen=True)
class ScheduleProvenance:
    """Provenance and bounded replay facts for the selected schedule."""

    schedule_path: str | None
    entries: int
    periods: tuple[int, ...]
    action_codes: tuple[str, ...]
    applied_in_fresh_model: bool


@dataclass(frozen=True)
class ScenarioReportRow:
    """One period of live inventory and applied-product results."""

    period: int
    harvested_area: float
    harvested_volume: float
    standing_volume: float


@dataclass(frozen=True)
class ScenarioReportResult:
    """Reviewable output from the bounded scenario report workflow."""

    ok: bool
    model_identity: ScenarioModelIdentity
    schedule_provenance: ScheduleProvenance
    initial_area: float | None
    initial_volume: float | None
    rows: tuple[ScenarioReportRow, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    source_model_files_unchanged: bool
    source_model_mutation_statement: str


def _identity(model_name: str, model_path: str) -> ScenarioModelIdentity:
    return ScenarioModelIdentity(
        model_name=model_name,
        model_path=model_path,
        base_year=None,
        horizon=None,
        period_length=None,
        nthemes=None,
        nactions=None,
        ndtypes=None,
    )


def _schedule_provenance(schedule_path: str | None = None) -> ScheduleProvenance:
    return ScheduleProvenance(
        schedule_path=schedule_path,
        entries=0,
        periods=(),
        action_codes=(),
        applied_in_fresh_model=False,
    )


def _source_hashes(model_dir: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(model_dir.iterdir()):
        if path.is_file():
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _resolve_inputs(inputs: ScenarioReportInputs) -> tuple[Path, Path]:
    if not isinstance(inputs.model_path, str) or not inputs.model_path.strip():
        raise ValueError('model_path is required')
    if not isinstance(inputs.model_name, str) or not inputs.model_name.strip():
        raise ValueError('model_name is required')
    if Path(inputs.model_name).name != inputs.model_name:
        raise ValueError('model_name must be a file base name, not a path')

    model_dir = Path(inputs.model_path).expanduser().resolve()
    if not model_dir.is_dir():
        raise ValueError(f'model_path is not a directory: {model_dir}')

    expected_schedule = model_dir / f'{inputs.model_name}.seq'
    if inputs.schedule_path:
        schedule_path = Path(inputs.schedule_path).expanduser()
        if not schedule_path.is_absolute():
            schedule_path = model_dir / schedule_path
        schedule_path = schedule_path.resolve()
    else:
        schedule_path = expected_schedule
    if schedule_path != expected_schedule:
        raise ValueError(
            'schedule_path must identify the selected model\'s sibling .seq file'
        )
    if not schedule_path.is_file():
        raise ValueError(f'schedule file does not exist: {schedule_path}')
    return model_dir, schedule_path


def _load_model(model_dir: Path, model_name: str) -> Any:
    from ws3.forest import ForestModel

    model = ForestModel(
        model_name=model_name,
        model_path=str(model_dir),
        base_year=2020,
        horizon=10,
        period_length=10,
        max_age=1000,
    )
    model.import_landscape_section()
    model.import_areas_section(convert_periods_to_years=10)
    model.import_yields_section(convert_periods_to_years=10)
    model.import_actions_section(convert_periods_to_years=10)
    model.import_transitions_section(convert_periods_to_years=10)
    model.reset_actions()
    return model


def _loaded_identity(model: Any, model_dir: Path) -> ScenarioModelIdentity:
    return ScenarioModelIdentity(
        model_name=str(model.model_name),
        model_path=str(model_dir),
        base_year=int(model.base_year),
        horizon=int(model.horizon),
        period_length=float(model.period_length),
        nthemes=int(model.nthemes()),
        nactions=len(model.actions),
        ndtypes=len(model.dtypes),
    )


def _failure(
    inputs: ScenarioReportInputs,
    error: str,
    *,
    model_path: Path | None = None,
    before: dict[str, str] | None = None,
) -> ScenarioReportResult:
    unchanged = True
    if model_path is not None and before is not None:
        unchanged = _source_hashes(model_path) == before
    schedule_path = None
    if model_path is not None:
        schedule_path = str(model_path / f'{inputs.model_name}.seq')
    statement = (
        'No source model file was mutated; the report failed before or during '
        'the isolated in-memory replay.'
        if unchanged
        else 'Source model file hashes changed; source-file mutation cannot be ruled out.'
    )
    return ScenarioReportResult(
        ok=False,
        model_identity=_identity(inputs.model_name, str(model_path or inputs.model_path)),
        schedule_provenance=_schedule_provenance(schedule_path),
        initial_area=None,
        initial_volume=None,
        rows=(),
        warnings=(),
        errors=(error,),
        source_model_files_unchanged=unchanged,
        source_model_mutation_statement=statement,
    )


def report_scenario_inventory_products(
    model_path: str | Path,
    model_name: str,
    schedule_path: str | Path | None = None,
) -> ScenarioReportResult:
    """Load a model, replay its bounded schedule, and report live results.

    The model is always newly loaded inside this call. The schedule is limited to
    the selected model's sibling ``.seq`` file, and no provider-generated actions
    or masks are accepted. Schedule application changes only this fresh in-memory
    model; source files are checked byte-for-byte before and after the replay.
    """
    inputs = ScenarioReportInputs(
        model_path=str(model_path),
        model_name=model_name,
        schedule_path=None if schedule_path is None else str(schedule_path),
    )
    model_dir: Path | None = None
    before: dict[str, str] | None = None
    try:
        model_dir, resolved_schedule = _resolve_inputs(inputs)
        before = _source_hashes(model_dir)
        model = _load_model(model_dir, inputs.model_name)
        identity = _loaded_identity(model, model_dir)
        schedule = model.import_schedule_section(convert_periods_to_years=10)
        invalid_periods = sorted({row[4] for row in schedule if row[4] not in model.periods})
        if invalid_periods:
            raise ValueError(
                f'schedule contains periods outside the model horizon: {invalid_periods}'
            )

        initial_area = float(model.inventory(0))
        initial_volume = float(model.inventory(0, 'totvol'))
        model.apply_schedule(
            schedule,
            max_period=model.horizon,
            fail_on_missingarea=True,
            compile_t_ycomps=True,
            compile_c_ycomps=True,
        )
        rows = tuple(
            ScenarioReportRow(
                period=period,
                harvested_area=float(
                    model.compile_product(period, '1.', acode='harvest')
                ),
                harvested_volume=float(
                    model.compile_product(period, 'totvol', acode='harvest')
                ),
                standing_volume=float(model.inventory(period, 'totvol')),
            )
            for period in model.periods
        )
        unchanged = _source_hashes(model_dir) == before
        warnings = ()
        if not schedule:
            warnings = (
                'The selected schedule contained no entries; harvested products '
                'are expected to be zero.',
            )
        elif not any(row.harvested_area > 0 or row.harvested_volume > 0 for row in rows):
            warnings = ('The applied schedule produced no harvest products.',)
        errors = () if unchanged else (
            'Source model file hashes changed during the report; result rejected.',
        )
        statement = (
            'No source model file was mutated; schedule application changed only '
            'the fresh in-memory ForestModel used for this report.'
            if unchanged
            else 'Source model file hashes changed; source-file mutation cannot be ruled out.'
        )
        return ScenarioReportResult(
            ok=not errors,
            model_identity=identity,
            schedule_provenance=ScheduleProvenance(
                schedule_path=str(resolved_schedule),
                entries=len(schedule),
                periods=tuple(sorted({row[4] for row in schedule})),
                action_codes=tuple(sorted({row[3] for row in schedule})),
                applied_in_fresh_model=True,
            ),
            initial_area=initial_area,
            initial_volume=initial_volume,
            rows=rows,
            warnings=warnings,
            errors=errors,
            source_model_files_unchanged=unchanged,
            source_model_mutation_statement=statement,
        )
    except Exception as exc:
        return _failure(
            inputs,
            f'{type(exc).__name__}: {exc}',
            model_path=model_dir,
            before=before,
        )


class ScenarioReport(Capability[ScenarioReportResult]):
    """MCP adapter for the deterministic scenario-report workflow.

    This adapter deliberately bypasses the provider retry loop. The host loads
    the model, applies only the bounded source schedule, and computes every
    reported number with WS3 APIs.
    """

    name = 'report_scenario_inventory_products'
    description = (
        'Replay the selected model\'s bounded .seq schedule in a fresh in-memory '
        'ForestModel and report initial inventory area, per-period harvested '
        'area/products, harvested volume/products, and standing volume. Validates '
        'the model and schedule paths, uses live inventory and compile_product '
        'calls, and verifies source model files remain unchanged. This is a '
        'host-side deterministic workflow; it does not accept provider-generated '
        'actions or masks.'
    )
    max_attempts = 1
    input_schema = {
        'type': 'object',
        'properties': {
            'model_path': {
                'type': 'string',
                'description': 'Directory containing the selected WS3 model files.',
            },
            'model_name': {
                'type': 'string',
                'description': 'Base name of the selected WS3 model.',
            },
            'schedule_path': {
                'type': 'string',
                'description': (
                    'Optional sibling .seq path; defaults to model_name.seq in '
                    'model_path.'
                ),
            },
        },
        'required': ['model_path', 'model_name'],
        'additionalProperties': False,
    }

    def from_payload(self, payload: dict[str, Any]) -> ScenarioReportInputs:
        """Build report inputs from an MCP payload."""
        return ScenarioReportInputs(
            model_path=str(payload.get('model_path', '')),
            model_name=str(payload.get('model_name', '')),
            schedule_path=(
                None
                if payload.get('schedule_path') in (None, '')
                else str(payload['schedule_path'])
            ),
        )

    def render(self, value: ScenarioReportResult) -> str:
        """Render the structured report as JSON for an MCP client."""
        return json.dumps(asdict(value), indent=2, sort_keys=True)

    def build_messages(
        self,
        inputs: ScenarioReportInputs,
        failures: tuple[str, ...],
    ) -> list[dict[str, str]]:
        """Satisfy the shared capability protocol; no provider call is used."""
        del inputs, failures
        return []

    def parse(self, raw: str) -> ScenarioReportResult:
        """Reject provider output because this workflow is host-side deterministic."""
        raise ParseError('scenario reports do not parse provider output')

    def validate(self, candidate: ScenarioReportResult, context: Any) -> Verdict:
        """Validate the direct workflow result when the shared protocol is used."""
        del context
        if isinstance(candidate, ScenarioReportResult):
            return Verdict.valid()
        return Verdict.invalid('scenario report returned an invalid result type')

    def run(
        self,
        inputs: ScenarioReportInputs | dict[str, Any],
        *,
        provider: Any = None,
        config: Any = None,
        context: Any = None,
        sink: Any = None,
    ) -> CapabilityResult[ScenarioReportResult]:
        """Run the host-side report without consulting the provider or context."""
        del provider, config, context, sink
        if isinstance(inputs, dict):
            inputs = self.from_payload(inputs)
        if not isinstance(inputs, ScenarioReportInputs):
            raise TypeError('scenario report inputs must be ScenarioReportInputs')
        report = report_scenario_inventory_products(
            inputs.model_path,
            inputs.model_name,
            inputs.schedule_path,
        )
        return CapabilityResult(
            ok=report.ok,
            value=report if report.ok else None,
            attempts=1,
            provenance_ids=(),
            errors=report.errors,
        )
