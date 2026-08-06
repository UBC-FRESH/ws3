"""
Validated, agent-backed capabilities for ws3.

Each capability pairs a prompt with a validator that checks the model's proposal
against **real ws3 state** -- resolving a mask against an actual
:py:class:`~ws3.forest.ForestModel`, confirming a symbol actually exists,
re-parsing a section that actually failed. Output that fails validation never
reaches the caller.

The shared machinery lives in ``fresh-agent-core``; this package contributes only
ws3's capabilities and their validators, because the validator is the part that
requires domain knowledge.

Install with::

    pip install ws3[agent]

Nothing here is imported by ``import ws3``. This module is optional, and importing
it without ``fresh-agent-core`` installed raises a clear error rather than an
opaque ``ModuleNotFoundError`` from three frames down.

IPython / Jupyter integration::

    In [1]: %load_ext ws3.agent.ipython_magics
    In [2]: %ws3_hint How do I add a fire disturbance?


Usage::

    import ws3.agent

    if ws3.agent.available():
        result = ws3.agent.run('build_mask', 'mature spruce stands', context=fm)
        if result.ok:
            print(result.value)
        else:
            print('no valid mask found:', result.errors)
"""

from __future__ import annotations

from typing import Any

__all__ = [
    'available',
    'build_model',
    'core_installed',
    'emit_actions',
    'emit_all',
    'emit_areas',
    'emit_landscape',
    'emit_outputs',
    'emit_transitions',
    'emit_yields',
    'get',
    'list_capabilities',
    'model_from_spec',
    'registry',
    'run',
]


def emit_landscape(spec: Any, output_dir: Any) -> Any:
    """Re-export from ws3.agent.emitter."""
    from ws3.agent.emitter import emit_landscape as _emit
    return _emit(spec, output_dir)


def emit_areas(spec: Any, output_dir: Any) -> Any:
    """Re-export from ws3.agent.emitter."""
    from ws3.agent.emitter import emit_areas as _emit
    return _emit(spec, output_dir)


def emit_yields(spec: Any, output_dir: Any) -> Any:
    """Re-export from ws3.agent.emitter."""
    from ws3.agent.emitter import emit_yields as _emit
    return _emit(spec, output_dir)


def emit_actions(spec: Any, output_dir: Any) -> Any:
    """Re-export from ws3.agent.emitter."""
    from ws3.agent.emitter import emit_actions as _emit
    return _emit(spec, output_dir)


def emit_outputs(spec: Any, output_dir: Any) -> Any:
    """Re-export from ws3.agent.emitter."""
    from ws3.agent.emitter import emit_outputs as _emit
    return _emit(spec, output_dir)


def emit_transitions(spec: Any, output_dir: Any) -> Any:
    """Re-export from ws3.agent.emitter."""
    from ws3.agent.emitter import emit_transitions as _emit
    return _emit(spec, output_dir)


def emit_all(spec: Any, output_dir: Any) -> Any:
    """Re-export from ws3.agent.emitter."""
    from ws3.agent.emitter import emit_all as _emit
    return _emit(spec, output_dir)


def model_from_spec(spec: Any, output_dir: Any) -> Any:
    """Build a ForestModel from a ModelSpec."""
    from ws3.agent.builder import ModelBuilder
    result = ModelBuilder(spec).build(output_dir)
    return result.model


def build_model(spec: Any, output_dir: Any) -> Any:
    """Build a ForestModel from a ModelSpec and return the full result."""
    from ws3.agent.builder import ModelBuilder
    return ModelBuilder(spec).build(output_dir)

try:
    import fresh_agent_core as _core
    _CORE_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - exercised via a test that fakes it
    _core = None  # type: ignore[assignment]
    _CORE_IMPORT_ERROR = exc


_INSTALL_HINT = (
    "ws3's agent capabilities require the 'fresh-agent-core' package.\n"
    '\n'
    'Install with:  pip install ws3[agent]\n'
    '\n'
    'This is an optional extra. Core ws3 modelling does not need it, and '
    '`import ws3` never loads it.'
)


def core_installed() -> bool:
    """
    True when ``fresh-agent-core`` is importable.

    Distinct from :py:func:`available`, which additionally requires a configured
    endpoint. Separated so a caller can tell "not installed" from "installed but
    unconfigured" -- they have different fixes.
    """
    return _core is not None


def _require_core() -> Any:
    """Return the core module, or raise with an actionable message."""
    if _core is None:
        raise ImportError(_INSTALL_HINT) from _CORE_IMPORT_ERROR
    return _core


def registry(fm: Any = None) -> Any:
    """
    The ws3 capability registry.

    Built lazily so that importing this module does not construct capabilities
    that may never be used.

    :param fm: Optional :py:class:`~ws3.forest.ForestModel`. Some capabilities
        need model state at *construction* time, not just at validation time --
        :py:class:`~ws3.agent.capabilities.build_mask.BuildMask` reads the theme
        count in order to assemble masks at the right arity.
    """
    _require_core()
    from ws3.agent.capabilities import build_registry
    return build_registry(fm)


def available() -> bool:
    """
    True when capabilities can actually run.

    Requires both that ``fresh-agent-core`` is installed and that an endpoint is
    configured. Never raises and never touches the network -- it answers "is this
    usable", not "is the endpoint reachable", so it is safe inside an ``if``.
    """
    if _core is None:
        return False
    return bool(_core.available())


def list_capabilities() -> list[dict[str, str]]:
    """
    Name and description for every ws3 capability.

    This is what an external agent reads to decide what to call. Each description
    states what the capability *validates*, so the caller knows what guarantee it
    is getting.
    """
    return list(registry().describe())


def _as_forest_model(context: Any) -> Any:
    """
    Return *context* if it looks like a :py:class:`~ws3.forest.ForestModel`.

    Duck-typed rather than isinstance-checked so that importing :py:mod:`ws3.agent`
    does not drag in :py:mod:`ws3.forest`. Capabilities take other kinds of
    context, so this filters rather than assumes.
    """
    return context if hasattr(context, 'nthemes') and hasattr(context, '_themes') else None


def get(name: str, fm: Any = None) -> Any:
    """
    Look up a capability by name.

    :param name: Capability name.
    :param fm: Optional model, passed to capabilities that need it at
        construction time.
    :raises KeyError: With the available names, since this is usually reached by
        an agent that guessed.
    """
    return registry(fm).get(name)


def run(
    name: str,
    inputs: Any,
    *,
    context: Any = None,
    provider: Any = None,
    config: Any = None,
    sink: Any = None,
) -> Any:
    """
    Run a capability by name.

    Convenience wrapper that resolves configuration and builds a provider. Pass
    ``provider`` explicitly to run offline against a
    :py:class:`~fresh_agent_core.FakeProvider`.

    :param name: Capability name.
    :param inputs: Capability-specific input.
    :param context: Real ws3 state for the validator to check against, e.g. a
        :py:class:`~ws3.forest.ForestModel`.
    :param provider: Override the model backend.
    :param config: Override the resolved configuration.
    :param sink: Where provenance records go.
    :return: A ``CapabilityResult``, which is either validated or explicitly
        unsuccessful.
    :raises AgentUnavailable: If no configuration can be resolved and no explicit
        provider was supplied.
    """
    core = _require_core()
    # Some capabilities need model state to build their prompt and to interpret
    # the response, not merely to validate it, so the context has to reach
    # construction as well as validation.
    capability = get(name, _as_forest_model(context))

    resolved = config if config is not None else core.config.resolve()
    if resolved is None:
        if provider is None:
            raise core.AgentUnavailable(
                'No agent configuration found. Set FRESH_AGENT_ENDPOINT and '
                'FRESH_AGENT_MODEL, or write ~/.config/fresh-agent/config.toml. '
                'Pass provider= explicitly to run offline against a FakeProvider.'
            )
        # A provider was supplied without configuration, which is the offline
        # testing path. Provenance still needs model/host metadata, so stand in
        # with an explicitly labelled placeholder rather than inventing values
        # that could be mistaken for a real endpoint.
        resolved = core.AgentConfig(endpoint='offline://test', model='offline')

    if provider is None:
        provider = core.OpenAIProvider(resolved)

    return capability.run(
        inputs,
        provider=provider,
        config=resolved,
        context=context,
        sink=sink,
    )
