"""
IPython line magics for ws3 agent capabilities.

Loads as an IPython extension::

    In [1]: %load_ext ws3.agent.ipython_magics

    In [2]: fm = ForestModel(...)
       ...: fm.import_landscape_section()
       ...: fm.import_areas_section()

    In [3]: %ws3_hint How do I add a partial fire disturbance to specific stands?

After loading, ``fm`` is discovered automatically from the user namespace.
No explicit model argument is needed.
"""

from __future__ import annotations

import re
from textwrap import dedent
from typing import Any

from fresh_agent_core import AgentConfig
from fresh_agent_core.config import resolve
from fresh_agent_core.provider import OpenAIProvider
from IPython import get_ipython  # type: ignore[attr-defined]
from IPython.core.magic import Magics, line_magic, magics_class, no_var_expand
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from IPython.display import Markdown, display

from ws3.agent.capabilities import build_registry
from ws3.agent.capabilities.build_mask import BuildMask, MaskRequest
from ws3.agent.capabilities.diagnose_import import DiagnoseImport, ImportFailure
from ws3.agent.capabilities.explain_exception import ExceptionReport, ExplainException
from ws3.agent.capabilities.inspect_model import InspectInputs, InspectModel
from ws3.agent.capabilities.rtfm_capability import RTFMCapability, RTFMInputs
from ws3.agent.capabilities.ws3_hint import HintInputs, Ws3Hint
from ws3.forest import ForestModel

__all__ = ['load_ipython_extension']


_QUESTION_MAGICS = (
    '%ws3_hint ',
    '%build_mask ',
    '%explain_exception ',
    '%rtfm ',
    '%ws3_inspect_model ',
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preserve_magic_questions(lines: list[str]) -> list[str]:
    """Keep natural-language magic questions out of IPython's ``obj?`` syntax."""
    transformed = []
    for line in lines:
        content = line.rstrip('\r\n')
        newline = line[len(content):]
        stripped = content.lstrip()
        if stripped.startswith(_QUESTION_MAGICS) and content.rstrip().endswith('?'):
            question_mark = content.rfind('?')
            content = content[:question_mark] + content[question_mark + 1:]
        transformed.append(content + newline)
    return transformed

def _find_fm(ipython: Any) -> object:
    """
    Discover the first ForestModel in the IPython user namespace.

    Uses ``isinstance`` against the actual ``ws3.forest.ForestModel`` type
    so subclasses are also matched. Raises RuntimeError if none is found.
    """
    ns = ipython.user_ns
    for _name, obj in ns.items():
        if isinstance(obj, ForestModel):
            return obj
    raise RuntimeError(
        "No ForestModel found in the IPython namespace. "
        "Create one first — e.g. ``fm = ForestModel(...)``."
    )


def _find_models(ipython: Any) -> list[tuple[str, object]]:
    """
    Enumerate every ForestModel in the IPython user namespace.

    Uses ``isinstance`` against the actual ``ws3.forest.ForestModel`` type
    so subclasses are also matched. Returns a list of
    ``(variable_name, model)`` tuples, ordered by insertion order of the
    namespace dict. The first entry matches what :py:func:`_find_fm` would
    return.
    """
    ns = ipython.user_ns
    found: list[tuple[str, object]] = []
    for name, obj in ns.items():
        if isinstance(obj, ForestModel):
            found.append((name, obj))
    return found


def _match_identifier(query: str, candidate: str) -> bool:
    """
    Check whether *query* is a complete identifier mention inside *candidate*.

    Both sides are normalised and split into tokens on word boundaries
    (``_``, ``-``, ``.``, and non-alphanumeric separators).  Matching requires
    the full candidate token to appear in the query -- accidental substrings
    like ``fm`` matching ``fm_alpha`` or ``spruce`` matching ``spruce_fir``
    are rejected so that multiple models never choose a wrong candidate.

    A blank query is not a match (the caller decides whether that is ``None``
    or ``True``).
    """
    if not query or not candidate:
        return False
    q_tokens = set(re.split(r'[\W_]+', query.lower()))
    c_tokens = set(re.split(r'[\W_]+', candidate.lower()))
    return bool(q_tokens) and c_tokens.issubset(q_tokens)


def _select_model(
    models: list[tuple[str, object]], query: str
) -> tuple[object | None, str | None]:
    """
    Pick one model from *models* given a free-text *query*.

    Returns ``(model, reason)`` where *reason* is one of:
    - ``None`` when a single unambiguous model is selected;
    - ``'ambiguous'`` when the query matches more than one model;
    - ``'no_match'`` when the query does not match any model.

    Multiple models are never silently resolved to one -- the caller must
    surface the ambiguity or refusal message.
    """
    query_lower = query.lower()
    matched: list[tuple[str, object]] = []
    for var_name, model in models:
        mv = getattr(model, 'model_name', '') or ''
        mname = getattr(model, 'name', '') or ''
        if (
            _match_identifier(query_lower, var_name)
            or _match_identifier(query_lower, mv)
            or _match_identifier(query_lower, mname)
        ):
            matched.append((var_name, model))

    if not matched:
        return None, 'no_match'
    if len(matched) > 1:
        return None, 'ambiguous'
    return matched[0][1], None


def _hint_context(fm: Any) -> str:
    """Summarize live action definitions for grounded modelling hints."""
    actions = getattr(fm, 'actions', {})
    if not actions:
        return (
            'The live ForestModel has no actions loaded. Do not invent an action; '
            'tell the user to import the ACTIONS and TRANSITIONS sections first.'
        )
    details = []
    for code in sorted(actions):
        action = actions[code]
        partial = bool(getattr(action, 'partial', []))
        details.append(f'{code} (partial definition: {partial})')
    return (
        'Live ForestModel action definitions: ' + ', '.join(details) + '. '
        'Use only these action codes. If the requested treatment is absent, say '
        'that it is not defined in this model instead of inventing an API.'
    )


def _make_config() -> AgentConfig:
    """Resolve the standard fresh-agent-core user configuration."""
    config = resolve()
    if config is None:
        raise RuntimeError(
            'No fresh-agent-core configuration found. Set FRESH_AGENT_ENDPOINT '
            'and FRESH_AGENT_MODEL or create ~/.config/fresh-agent/config.toml.'
        )
    return config


def _make_provider(config: AgentConfig) -> OpenAIProvider:
    """Build an OpenAI-compatible provider for the capability calls."""
    return OpenAIProvider(config)


def _fmt_verdict(cap_name: str, result: Any) -> str:
    """Format a capability result as notebook-friendly Markdown."""
    heading = cap_name.replace('_', ' ').title().replace('Ws3', 'WS3')

    # --- InspectResult: deterministic metadata snapshot of a ForestModel ---
    if cap_name == 'inspect_model':
        v = getattr(result, 'value', None)
        errors = getattr(result, 'errors', None) or ()
        ok = getattr(result, 'ok', False)
        if not ok or v is None:
            if errors:
                return (
                    f'### {heading} rejected\n\n' +
                    '\n'.join(f'- {e}' for e in errors)
                )
            return f'### {heading}\n\nNo value returned.'
        if v.unsupported:
            return f'### {heading}\n\n**Unsupported query**: {v.unsupported}'
        lines = [f'### {heading}', '']
        _inspect_labels = [
            ('model_name', 'model_name'),
            ('name', 'name'),
            ('base_year', 'base_year'),
            ('horizon', 'horizon'),
            ('period_length', 'period_length'),
            ('periods', 'periods'),
            ('nthemes', 'nthemes'),
            ('nactions', 'nactions'),
            ('ndtypes', 'ndtypes'),
            ('total_area (period 1)', 'total_area'),
        ]
        for label, attr in _inspect_labels:
            val = getattr(v, attr, None)
            if val is not None:
                lines.append(f'- **{label}**: `{val}`')
            elif attr == 'total_area':
                lines.append(f'- **{label}**: unavailable')
        lines.extend([
            '',
            '> Read-only snapshot. Values come from the live ForestModel.',
            '> The provider selected the bounded operation; the executor '
            'computed the numeric values directly.',
            '> Unsupported requests return an explicit unsupported result '
            'rather than fabricated values.',
        ])
        return '\n'.join(lines)

    if result.ok:
        val = result.value
        lines = [f'### {heading}']
        if hasattr(val, 'hint') and val.hint:
            lines.extend(['', val.hint])
        if hasattr(val, 'suggested_steps') and val.suggested_steps:
            lines.extend(['', '#### Suggested steps', ''])
            for i, s in enumerate(val.suggested_steps, 1):
                lines.append(f'{i}. {s}')
        if hasattr(val, 'mask') and val.mask:
            lines.extend(['', '#### Validated mask', '', f'`{val.mask}`'])
        if hasattr(val, 'fix') and val.fix:
            lines.extend([
                '',
                '#### Suggested fix',
                '',
                '```text',
                dedent(val.fix).strip(),
                '```',
            ])
        if hasattr(val, 'cause') and val.cause:
            lines.extend(['', '#### Cause', '', f'> {val.cause}'])
        if getattr(val, 'rtfm_footer', None):
            lines.extend(['', '---', '', val.rtfm_footer])
        return '\n'.join(lines)

    lines = [f'### {heading} rejected', '']
    lines.extend(f'- {error}' for error in result.errors)
    return '\n'.join(lines)


def _display_verdict(cap_name: str, result: Any) -> None:
    """Display a capability result without producing a quoted ``Out`` value."""
    display(Markdown(_fmt_verdict(cap_name, result)))  # type: ignore[no-untyped-call]


# ---------------------------------------------------------------------------
# Magics
# ---------------------------------------------------------------------------

@magics_class  # type: ignore[misc]
class Ws3Magics(Magics):  # type: ignore[misc]

    @line_magic  # type: ignore[untyped-decorator]
    @no_var_expand  # type: ignore[untyped-decorator]
    @magic_arguments()  # type: ignore[no-untyped-call, untyped-decorator]
    @argument('goal', type=str, nargs='*', help='Goal or error description')  # type: ignore[no-untyped-call, untyped-decorator]
    def ws3_hint(self, line: str) -> str | None:
        """
        General ws3 modelling guidance with verifiable symbol and URL references.

        Usage::

            %ws3_hint How do I add a fire disturbance to my model?
        """
        args = parse_argstring(self.ws3_hint, line)  # type: ignore[no-untyped-call]
        if not args.goal:
            return "Usage: %ws3_hint <your question>"
        goal = ' '.join(args.goal)

        fm = _find_fm(self.shell)
        config = _make_config()
        cap = Ws3Hint()
        result = cap.run(
            HintInputs(goal=goal, context=_hint_context(fm)),
            provider=_make_provider(config),
            config=config,
            context=fm,
        )
        _display_verdict('ws3_hint', result)
        return None

    @line_magic  # type: ignore[untyped-decorator]
    @no_var_expand  # type: ignore[untyped-decorator]
    @magic_arguments()  # type: ignore[no-untyped-call, untyped-decorator]
    @argument('description', type=str, nargs='+', help='Mask description')  # type: ignore[no-untyped-call, untyped-decorator]
    def build_mask(self, line: str) -> str | None:
        """
        Build a ws3 development-type mask from a natural-language description.

        The mask is validated against the live ForestModel (``fm``) to guarantee
        it matches at least one real development type.

        Usage::

            %build_mask all mature spruce stands
        """
        args = parse_argstring(self.build_mask, line)  # type: ignore[no-untyped-call]
        if not args.description:
            return "Usage: %build_mask <mask description>"

        fm = _find_fm(self.shell)
        config = _make_config()
        cap = BuildMask(fm)
        result = cap.run(
            MaskRequest(description=' '.join(args.description)),
            provider=_make_provider(config),
            config=config,
            context=fm,
        )
        _display_verdict('build_mask', result)
        return None

    @line_magic  # type: ignore[untyped-decorator]
    @no_var_expand  # type: ignore[untyped-decorator]
    @magic_arguments()  # type: ignore[no-untyped-call, untyped-decorator]
    @argument('error_text', type=str, nargs='*', help='Error or traceback text')  # type: ignore[no-untyped-call, untyped-decorator]
    def explain_exception(self, line: str) -> str | None:
        """
        Explain a ws3 Python exception in plain language.

        Usage::

            %explain_exception KeyError: 'theme not found'
        """
        args = parse_argstring(self.explain_exception, line)  # type: ignore[no-untyped-call]
        if not args.error_text:
            return "Usage: %explain_exception <error message or traceback>"

        fm = _find_fm(self.shell)
        config = _make_config()
        cap = ExplainException()
        result = cap.run(
            ExceptionReport(
                exc_type='Exception',
                message=' '.join(args.error_text),
                traceback_text='',
                context='',
            ),
            provider=_make_provider(config),
            config=config,
            context=fm,
        )
        _display_verdict('explain_exception', result)
        return None

    @line_magic  # type: ignore[untyped-decorator]
    @no_var_expand  # type: ignore[untyped-decorator]
    @magic_arguments()  # type: ignore[no-untyped-call, untyped-decorator]
    @argument('section', type=str, nargs='?', default='', help='Section name (optional)')  # type: ignore[no-untyped-call, untyped-decorator]
    @argument('model_path', type=str, nargs='?', default='', help='Path to model (optional)')  # type: ignore[no-untyped-call, untyped-decorator]
    def diagnose_import(self, line: str) -> str | None:
        """
        Diagnose a Woodstock section import failure and suggest a corrected line.

        Usage::

            %diagnose_import
            %diagnose_import landscape /path/to/model
        """
        args = parse_argstring(self.diagnose_import, line)  # type: ignore[no-untyped-call]
        fm = _find_fm(self.shell)
        config = _make_config()
        cap = DiagnoseImport()
        failure = ImportFailure(
            model_path=args.model_path or '',
            model_name=getattr(fm, 'model_name', ''),
            section=args.section or '',
            error='',
            excerpt='',
        )
        result = cap.run(
            failure,
            provider=_make_provider(config),
            config=config,
            context=failure,
        )
        _display_verdict('diagnose_import', result)
        return None

    @line_magic  # type: ignore[untyped-decorator]
    @no_var_expand  # type: ignore[untyped-decorator]
    @magic_arguments()  # type: ignore[no-untyped-call, untyped-decorator]
    @argument('query', type=str, nargs='*', help='Goal or error description')  # type: ignore[no-untyped-call, untyped-decorator]
    def rtfm(self, line: str) -> str | None:
        """
        Route a goal or error to the correct ws3 capability.

        Usage::

            %rtfm How do I add a new species?
            %rtfm KeyError: 'theme not found'
        """
        args = parse_argstring(self.rtfm, line)  # type: ignore[no-untyped-call]
        if not args.query:
            return "Usage: %rtfm <your question or error>"

        fm = _find_fm(self.shell)
        config = _make_config()
        cap = RTFMCapability()
        result = cap.run(
            RTFMInputs(query=' '.join(args.query)),
            provider=_make_provider(config),
            config=config,
            context=fm,
        )
        _display_verdict('rtfm', result)
        return None

    @line_magic  # type: ignore[untyped-decorator]
    @no_var_expand  # type: ignore[untyped-decorator]
    @magic_arguments()  # type: ignore[no-untyped-call, untyped-decorator]
    @argument('query', type=str, nargs='*', help='Query (optional)')  # type: ignore[no-untyped-call, untyped-decorator]
    def ws3_inspect_model(self, line: str) -> None:
        """
        Show a read-only metadata snapshot of the live ForestModel.

        Displays model_name, base_year, horizon, period_length, periods,
        theme/action/dtype counts, and total area (period 1 only).

        Usage::

            %ws3_inspect_model
            %ws3_inspect_model full snapshot
        """
        args = parse_argstring(self.ws3_inspect_model, line)  # type: ignore[no-untyped-call]
        query = ' '.join(args.query) if args.query else 'full snapshot'

        # Enumerate all ForestModels
        models = _find_models(self.shell)

        if not models:
            display(Markdown(  # type: ignore[no-untyped-call]
                '### WS3 Inspect Model\n\n'
                'No ForestModel found in the IPython namespace. '
                "Create one first — e.g. ``fm = ForestModel(...)``."
            ))
            return

        if len(models) == 1:
            fm = models[0][1]
        else:
            # Multiple models: require explicit query with a model identifier.
            # _select_model uses complete-identifier matching so substrings
            # like ``fm`` in ``fm_alpha`` or ``spruce`` in ``spruce_fir``
            # never silently select a wrong candidate.
            fm, reason = _select_model(models, query)
            if reason == 'no_match':
                # List candidates and refuse to silently pick
                lines = ['### WS3 Inspect Model — multiple models found', '']
                for var_name, model in models:
                    mn = getattr(model, 'model_name', '?') or '?'
                    mname = getattr(model, 'name', '?') or '?'
                    lines.append(
                        f'- ``{var_name}`` (model_name=``{mn}``, name=``{mname}``)'
                    )
                lines.extend([
                    '',
                    'Specify which model to inspect by including its variable '
                    'name or model_name in the query, e.g.::',
                    '',
                    '    %ws3_inspect_model fm',
                    '    %ws3_inspect_model my_model_name',
                ])
                display(Markdown('\n'.join(lines)))  # type: ignore[no-untyped-call]
                return
            if reason == 'ambiguous':
                lines = ['### WS3 Inspect Model — ambiguous query', '']
                for var_name, model in models:
                    mn = getattr(model, 'model_name', '?') or '?'
                    mname = getattr(model, 'name', '?') or '?'
                    lines.append(
                        f'- ``{var_name}`` (model_name=``{mn}``, name=``{mname}``)'
                    )
                lines.extend([
                    '',
                    f'The query ``{query!r}`` matches more than one model. '
                    'Be more specific by including the full variable name, '
                    'model_name, or public name of the model you want to inspect.',
                ])
                display(Markdown('\n'.join(lines)))  # type: ignore[no-untyped-call]
                return
            # reason is None -> exactly one match
            assert fm is not None

        config = _make_config()
        cap = InspectModel()
        inputs = InspectInputs(query=query)
        result = cap.run(
            inputs,
            provider=_make_provider(config),
            config=config,
            context=fm,
        )

        if result.ok and result.value is not None:
            _display_verdict('inspect_model', result)
        else:
            display(Markdown(  # type: ignore[no-untyped-call]
                '### WS3 Inspect Model rejected\n' +
                '\n'.join(f'- {e}' for e in result.errors)
            ))

    @line_magic  # type: ignore[untyped-decorator]
    def ws3_capabilities(self, line: str) -> None:
        """
        List all available ws3 agent capabilities.

        Usage::

            %ws3_capabilities
        """
        fm = _find_fm(self.shell)
        registry = build_registry(fm)
        lines = ['## Available ws3 capabilities', '']
        for cap in registry:
            lines.append(f'- **{cap.name}**: {cap.description}')
        lines.append('')
        lines.append(
            "Use `%ws3_hint <question>` for general modelling guidance, or\n"
            "`%build_mask`, `%explain_exception`, `%diagnose_import`, `%rtfm`, "
            "`%ws3_inspect_model` for specific tasks."
        )
        display(Markdown('\n'.join(lines)))  # type: ignore[no-untyped-call]


def load_ipython_extension(ipython: Any | None = None) -> None:
    """Register the ws3 magics with IPython."""
    ipython = ipython or get_ipython()  # type: ignore[no-untyped-call]
    if _preserve_magic_questions not in ipython.input_transformers_cleanup:
        ipython.input_transformers_cleanup.append(_preserve_magic_questions)
    ipython.register_magics(Ws3Magics)


def unload_ipython_extension(ipython: Any) -> None:
    """Remove the ws3 input transform when unloading the extension."""
    if _preserve_magic_questions in ipython.input_transformers_cleanup:
        ipython.input_transformers_cleanup.remove(_preserve_magic_questions)
