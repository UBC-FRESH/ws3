"""
Builder/adapter for constructing ForestModel instances from ModelSpec.

This module provides :py:class:`ModelBuilder`, which takes a
:py:class:`ws3.agent.spec.ModelSpec`, emits Woodstock section files to an
explicit directory, and imports a fresh :py:class:`~ws3.forest.ForestModel`.

The builder never mutates an existing model. It always creates a new
ForestModel instance, imports the emitted files, and returns it.

Period handling: the builder applies the period-to-year conversion when
importing, using the spec's ``period_length``. All ages in the spec are
interpreted as periods and converted to years during import.

Transition data with unsupported features (e.g. ``theme_append``, or
``theme_replace`` not using the ``_TH`` prefix) is rejected before emission.
Action codes are accepted without type classification; action fields that are
not emitted are reported in the :py:class:`BuildResult.loss` dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ws3.agent.emitter import emit_all
from ws3.agent.spec import (
    ModelSpec,
    ModelSpecError,
    UnsupportedTransitionError,
)


@dataclass(frozen=True)
class BuildResult:
    """
    Result of building a ForestModel from a ModelSpec.

    :param model: The imported ForestModel.
    :param output_dir: Directory containing the emitted section files.
    :param emitted_paths: Mapping of section name to file path.
    :param period_length: Period length in years (from the spec).
    :param loss: JSON-serializable record of features not supported by
        this slice. Empty when no features were dropped or rejected.
    """

    model: Any
    output_dir: Path
    emitted_paths: dict[str, Path]
    period_length: int
    loss: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'model_name': self.model.model_name,
            'output_dir': str(self.output_dir),
            'emitted_paths': {k: str(v) for k, v in self.emitted_paths.items()},
            'period_length': self.period_length,
            'n_development_types': len(self.model.dtypes),
            'n_actions': len(self.model.actions),
            'loss': self.loss,
        }


class ModelBuilder:
    """
    Build a ForestModel from a ModelSpec.

    The builder:

    1. Validates the spec (already done in __post_init__, but re-checks here).
     2. Rejects unsupported transition data and records unsupported action
         metadata in the result loss report.
    3. Emits Woodstock section files to an explicit directory.
    4. Imports a fresh ForestModel from the emitted files.
    5. Returns the result.

    The builder never mutates an existing model. It always creates a new
    ForestModel instance.

    :param spec: The model specification.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def build(
        self,
        output_dir: Path | str,
        *,
        overwrite: bool = False,
    ) -> BuildResult:
        """
        Build a ForestModel from the spec.

        :param output_dir: Directory to emit section files to. If the directory
            does not exist, it is created.
        :param overwrite: If ``True``, allow writing into a non-empty
            ``output_dir``. Defaults to ``False``; a non-empty output directory
            raises :py:class:`ModelSpecError`. Never silently deletes or
            overwrites unrelated files.
        :return: A :py:class:`BuildResult` with the imported model and paths.
        :raises UnsupportedTransitionError: If the spec contains unsupported transitions.
        :raises ModelSpecError: If the spec is invalid, the model_name could
            escape ``output_dir``, or ``output_dir`` is non-empty and
            ``overwrite`` is ``False``.
        """
        output_dir = Path(output_dir)

        # Validate unsupported features before emission (side-effect free).
        self._check_unsupported()

        # Validate model_name cannot escape output_dir.
        self._validate_model_name()

        # Validate output_dir is safe to write into.
        self._validate_output_dir(output_dir, overwrite=overwrite)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Emit section files.
        emitted_paths = emit_all(self.spec, output_dir)

        # Import a fresh ForestModel.
        from ws3.forest import ForestModel

        model = ForestModel(
            model_name=self.spec.model_name,
            model_path=str(output_dir),
            base_year=self.spec.base_year,
            horizon=self.spec.horizon,
            period_length=self.spec.period_length,
            max_age=self.spec.max_age,
        )

        # Import landscape section.
        if self.spec.themes:
            model.import_landscape_section()

        # Import areas section with period-to-year conversion.
        if self.spec.areas:
            model.import_areas_section(convert_periods_to_years=self.spec.period_length)

        # Import yields section with period-to-year conversion.
        if self.spec.yields:
            model.import_yields_section(convert_periods_to_years=self.spec.period_length)

        # Import actions section with period-to-year conversion.
        if self.spec.actions:
            model.import_actions_section(convert_periods_to_years=self.spec.period_length)

        # Import transitions section with period-to-year conversion.
        if self.spec.transitions:
            model.import_transitions_section(convert_periods_to_years=self.spec.period_length)

        # Import outputs section.
        if self.spec.outputs:
            model.import_outputs_section()

        loss = self._compute_loss()

        return BuildResult(
            model=model,
            output_dir=output_dir,
            emitted_paths=emitted_paths,
            period_length=self.spec.period_length,
            loss=loss,
        )

    def _check_unsupported(self) -> None:
        """Validate that transitions use only supported features.

        Transition emission is supported for theme_replace (with _TH prefix)
        and theme_mask features. Unsupported features like theme_append and
        theme_replace without the _TH prefix are rejected before emission.

        Action type validation is not performed here. Action fields that are
        not emitted are reported later in the BuildResult.loss dict.
        """
        # Validate transition features.
        for case, trans in self.spec.transitions.items():
            if trans.theme_append:
                raise UnsupportedTransitionError(
                    f'Transition {case} has unsupported theme_append: '
                    f'{trans.theme_append!r}'
                )
            if trans.theme_replace and not trans.theme_replace.startswith('_TH'):
                raise UnsupportedTransitionError(
                    f'Transition {case} has unsupported theme_replace: '
                    f'{trans.theme_replace!r}'
                )

    def _compute_loss(self) -> dict[str, Any]:
        """Compute loss report for features not supported by this slice.

        Returns a dict describing which features were dropped or rejected.
        Empty when no features were dropped.
        """
        loss: dict[str, Any] = {}
        # Check for unsupported action features.
        for acode, action in self.spec.actions.items():
            if action.target_age is not None:
                loss.setdefault('actions', []).append(
                    f'{acode}: target_age is not supported'
                )
            if action.lock_exempt:
                loss.setdefault('actions', []).append(
                    f'{acode}: lock_exempt is not supported'
                )
            if action.description:
                loss.setdefault('actions', []).append(
                    f'{acode}: description is not imported'
                )
        # Check for unsupported transition features.
        for case, trans in self.spec.transitions.items():
            if trans.theme_mask:
                loss.setdefault('transitions', []).append(
                    f'{case}: theme_mask is not imported'
                )
        return loss

    def _validate_model_name(self) -> None:
        """Reject model_name values that could escape output_dir.

        Unsafe names include path separators, ``..`` components, empty
        strings, or names that resolve to a path outside the intended
        output directory.
        """
        name = self.spec.model_name
        if not name or not name.strip():
            raise ModelSpecError(
                f'model_name is empty or blank: {name!r}'
            )
        # Reject any path separator characters.
        if '/' in name or '\\' in name:
            raise ModelSpecError(
                f'model_name contains path separators: {name!r}'
            )
        # Reject '..' path traversal components.
        parts = name.replace('\\', '/').split('/')
        if '..' in parts:
            raise ModelSpecError(
                f'model_name contains path traversal (..): {name!r}'
            )
        # Reject names that are just whitespace after stripping.
        if name != name.strip():
            raise ModelSpecError(
                f'model_name has leading/trailing whitespace: {name!r}'
            )

    def _validate_output_dir(
        self, output_dir: Path, *, overwrite: bool = False
    ) -> None:
        """Reject a non-empty output_dir unless overwrite is explicitly True.

        :param output_dir: The resolved output directory path.
        :param overwrite: Whether to allow writing into a non-empty directory.
        :raises ModelSpecError: If the directory is non-empty and overwrite is
            False.
        """
        if not output_dir.exists():
            return
        if overwrite:
            return
        entries = list(output_dir.iterdir())
        if entries:
            sample = ', '.join(repr(e.name) for e in entries[:5])
            raise ModelSpecError(
                f'output_dir is non-empty and overwrite=False: '
                f'{output_dir} (sample: {sample})'
            )
