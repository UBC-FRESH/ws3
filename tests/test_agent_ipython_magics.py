"""Regression tests for the ws3 IPython magic integration."""

from types import SimpleNamespace

import pytest
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import Markdown

from ws3.agent import ipython_magics
from ws3.agent.ipython_magics import Ws3Magics, load_ipython_extension
from ws3.forest import ForestModel


@pytest.mark.parametrize(
    ('magic_name', 'line'),
    [
        ('ws3_hint', 'How do I use {disturbance}?'),
        ('build_mask', 'stands affected by {disturbance}'),
        ('explain_exception', 'KeyError: {disturbance}'),
        ('diagnose_import', '{disturbance}'),
        ('rtfm', 'How do I use {disturbance}?'),
    ],
)
def test_agent_magics_preserve_literal_braces(magic_name, line):
    shell = InteractiveShell()
    shell.register_magics(Ws3Magics)

    with pytest.raises(RuntimeError, match='No ForestModel'):
        shell.run_line_magic(magic_name, line)


def test_ws3_hint_question_is_not_rewritten_as_object_help():
    shell = InteractiveShell()
    load_ipython_extension(shell)
    source = '%ws3_hint How do I add a fire disturbance?'

    transformed = shell.transform_cell(source)
    assert "run_line_magic('ws3_hint'" in transformed
    assert "run_line_magic('pinfo'" not in transformed

    result = shell.run_cell(source, store_history=False)
    assert isinstance(result.error_in_exec, RuntimeError)
    assert 'No ForestModel' in str(result.error_in_exec)


def test_verdict_is_structured_markdown():
    value = SimpleNamespace(
        hint='Define the disturbance in the model before applying it.',
        suggested_steps=['Add an action.', 'Add its transition.'],
        rtfm_footer='**References**\n\n- https://example.test/ws3',
    )
    result = SimpleNamespace(ok=True, value=value, errors=[])

    rendered = ipython_magics._fmt_verdict('ws3_hint', result)

    assert rendered.startswith('### WS3 Hint')
    assert '#### Suggested steps' in rendered
    assert '1. Add an action.' in rendered
    assert '\n---\n' in rendered


def test_display_verdict_uses_markdown_and_returns_none(monkeypatch):
    displayed = []
    monkeypatch.setattr(ipython_magics, 'display', displayed.append)
    result = SimpleNamespace(
        ok=False,
        value=None,
        errors=['No validated answer was produced.'],
    )

    returned = ipython_magics._display_verdict('ws3_hint', result)

    assert returned is None
    assert len(displayed) == 1
    assert isinstance(displayed[0], Markdown)
    assert 'WS3 Hint rejected' in displayed[0].data


def test_make_config_uses_standard_fresh_agent_core_resolver(monkeypatch):
    expected = SimpleNamespace(endpoint='https://agent.example.test', model='standard-model')
    monkeypatch.setattr(ipython_magics, 'resolve', lambda: expected)

    assert ipython_magics._make_config() is expected


def test_diagnose_import_passes_failure_as_validator_context(monkeypatch):
    fm = _fake_fm('fm', model_name='example')
    calls = []
    monkeypatch.setattr(ipython_magics, '_find_fm', lambda shell: fm)
    monkeypatch.setattr(ipython_magics, '_make_config', lambda: object())
    monkeypatch.setattr(ipython_magics, '_make_provider', lambda config: object())
    monkeypatch.setattr(ipython_magics, '_display_verdict', lambda name, result: None)

    def run(self, inputs, *, provider, config, context):
        calls.append((inputs, context))
        return SimpleNamespace(ok=True, value=None, errors=[])

    monkeypatch.setattr(ipython_magics.DiagnoseImport, 'run', run)
    shell = InteractiveShell()
    shell.register_magics(Ws3Magics)

    shell.run_line_magic('diagnose_import', 'landscape /tmp/model')

    assert len(calls) == 1
    inputs, context = calls[0]
    assert context is inputs
    assert inputs.model_name == 'example'
    assert inputs.section == 'landscape'
    assert inputs.model_path == '/tmp/model'


def test_ws3_capabilities_displays_markdown_and_returns_none(monkeypatch):
    """Regression: %ws3_capabilities must render structured Markdown and return None."""
    displayed = []
    monkeypatch.setattr(ipython_magics, 'display', displayed.append)

    fake_caps = [
        SimpleNamespace(name='ws3_hint', description='General modelling guidance.'),
        SimpleNamespace(name='build_mask', description='Build a development-type mask.'),
    ]

    class FakeRegistry:
        def __iter__(self):
            return iter(fake_caps)

    monkeypatch.setattr(
        ipython_magics, 'build_registry', lambda fm: FakeRegistry()
    )

    # Bypass _find_fm entirely.
    monkeypatch.setattr(ipython_magics, '_find_fm', lambda ipython: SimpleNamespace())

    shell = InteractiveShell()
    ipython_magics.load_ipython_extension(shell)

    result = shell.run_line_magic('ws3_capabilities', '')

    assert result is None
    assert len(displayed) == 1
    assert isinstance(displayed[0], Markdown)
    md_text = displayed[0].data
    assert '## Available ws3 capabilities' in md_text
    assert '**ws3_hint**' in md_text
    assert '**build_mask**' in md_text
    assert '%ws3_hint' in md_text


# ---------------------------------------------------------------------------
# Phase 8 — `%ws3_inspect_model` regression coverage
# ---------------------------------------------------------------------------


class _FakeForestModel(ForestModel):
    """Lightweight ``ForestModel`` subclass for testing — bypasses ``__init__``."""


def _fake_fm(var_name: str, model_name: str = '', name: str = '') -> object:
    """Build a real ``ForestModel`` subclass stand-in (no heavyweight ``__init__``)."""

    fm = _FakeForestModel.__new__(_FakeForestModel)
    fm.namespace_key = var_name
    fm.model_name = model_name or var_name
    fm.name = name or model_name or var_name
    fm.base_year = 2020
    fm.period_length = 10
    fm.periods = list(range(1, 11))
    fm._themes = {'th_a': None, 'th_b': None}
    fm.actions = ['cut', 'thin']
    fm.dtypes = {'dt_1': None, 'dt_2': None, 'dt_3': None}

    class _FakeDtype:
        def area(self, _p: int) -> float:
            return 12.5

    fm.dtypes = {'dt_1': _FakeDtype(), 'dt_2': _FakeDtype(), 'dt_3': _FakeDtype()}
    return fm


class TestInspectModelQuestionPreservation:
    def test_inspect_model_trailing_question_mark_not_rewritten(
        self,
    ):
        """A terminal ``?`` on ``%ws3_inspect_model`` stays as a magic call, not ``pinfo``."""
        shell = InteractiveShell()
        load_ipython_extension(shell)
        source = '%ws3_inspect_model show me the model metadata?'

        transformed = shell.transform_cell(source)
        assert "run_line_magic('ws3_inspect_model'" in transformed
        assert 'run_line_magic(\'pinfo\'' not in transformed

    def test_inspect_model_no_space_question_mark_is_not_supported(
        self,
    ):
        """Without a space before ``?`` the transformer leaves the call alone;
        this mirrors ``_QUESTION_MAGICS`` which requires a trailing space."""
        shell = InteractiveShell()
        load_ipython_extension(shell)
        source = '%ws3_inspect_model?'

        transformed = shell.transform_cell(source)
        # The no-space form is not in ``_QUESTION_MAGICS``, so IPython rewrites
        # it to a ``pinfo`` call. The assertion below guards the documented
        # behaviour: the operator must put a space before ``?``.
        assert 'run_line_magic(\'pinfo\'' in transformed


class TestInspectModelFmtVerdict:
    def test_fmt_verdict_inspect_model_renders_identity(self):
        """``_fmt_verdict('inspect_model', ...)`` renders the metadata as Markdown."""
        from ws3.agent.capabilities.inspect_model import InspectResult

        v = InspectResult(
            model_name='tsa24_clipped',
            name='tsa24_clipped',
            base_year=2020,
            horizon=10,
            period_length=10.0,
            periods=list(range(1, 11)),
            nthemes=2,
            nactions=2,
            ndtypes=3,
            total_area=37.5,
        )
        result = SimpleNamespace(ok=True, value=v, errors=[])

        rendered = ipython_magics._fmt_verdict('inspect_model', result)

        assert rendered.startswith('### Inspect Model')
        assert '- **model_name**: `tsa24_clipped`' in rendered
        assert '- **base_year**: `2020`' in rendered
        assert '- **horizon**: `10`' in rendered
        assert '- **periods**: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`' in rendered
        assert '- **nthemes**: `2`' in rendered
        assert '- **nactions**: `2`' in rendered
        assert '- **ndtypes**: `3`' in rendered
        assert '- **total_area (period 1)**: `37.5`' in rendered
        assert (
            '> Read-only snapshot. Values come from the live ForestModel.'
            in rendered
        )

    def test_fmt_verdict_inspect_model_marks_unsupported_fields(self):
        """``total_area`` is shown as unavailable when the underlying dt is None."""
        from ws3.agent.capabilities.inspect_model import InspectResult

        v = InspectResult(
            model_name='bare',
            name='bare',
            base_year=None,
            horizon=None,
            period_length=None,
            periods=None,
            nthemes=0,
            nactions=0,
            ndtypes=0,
            total_area=None,
        )
        result = SimpleNamespace(ok=True, value=v, errors=[])
        rendered = ipython_magics._fmt_verdict('inspect_model', result)

        assert '- **total_area (period 1)**: unavailable' in rendered
        assert '- **base_year**: `None`' not in rendered

    def test_fmt_verdict_inspect_model_reports_unsupported_query(self):
        """An unsupported query is surfaced plainly rather than silently dropped."""
        from ws3.agent.capabilities.inspect_model import InspectResult

        v = InspectResult(
            model_name='tsa24_clipped',
            name='tsa24_clipped',
            base_year=2020,
            horizon=10,
            period_length=10.0,
            periods=list(range(1, 11)),
            nthemes=1,
            nactions=1,
            ndtypes=1,
            total_area=10.0,
            unsupported='Plotting is not supported.',
        )
        result = SimpleNamespace(ok=True, value=v, errors=[])
        rendered = ipython_magics._fmt_verdict('inspect_model', result)

        assert '**Unsupported query**: Plotting is not supported.' in rendered

    def test_fmt_verdict_inspect_model_rejected_lists_errors(self):
        """A rejected result renders each error as a bullet."""
        result = SimpleNamespace(
            ok=False,
            value=None,
            errors=['field `nthemes` was missing', 'field `periods` raised'],
        )
        rendered = ipython_magics._fmt_verdict('inspect_model', result)
        assert rendered.startswith('### Inspect Model rejected')
        assert '- field `nthemes` was missing' in rendered
        assert '- field `periods` raised' in rendered


class TestInspectModelMagic:
    """
    End-to-end run of ``%ws3_inspect_model`` through a real InteractiveShell
    with the ForestModel lookup and the capability provider stubbed.
    """

    def _build_shell_with_fm(self, models: list[object]) -> InteractiveShell:
        shell = InteractiveShell()
        load_ipython_extension(shell)
        for m in models:
            ns_key = getattr(m, 'namespace_key', m.model_name)
            shell.user_ns[ns_key] = m
        return shell

    def _make_fake_result(self):
        from ws3.agent.capabilities.inspect_model import InspectResult

        return SimpleNamespace(
            ok=True,
            value=InspectResult(
                model_name='tsa24_clipped',
                name='tsa24_clipped',
                base_year=2020,
                horizon=10,
                period_length=10.0,
                periods=list(range(1, 11)),
                nthemes=2,
                nactions=2,
                ndtypes=3,
                total_area=37.5,
            ),
            errors=[],
        )

    def test_one_model_displays_markdown_and_returns_none(self, monkeypatch):
        """With exactly one model, the magic displays Markdown and returns None."""
        from ws3.agent import ipython_magics as mod

        displayed = []
        monkeypatch.setattr(mod, 'display', displayed.append)

        result = self._make_fake_result()
        monkeypatch.setattr(
            mod.InspectModel, 'run', lambda self_inst, *a, **kw: result
        )

        shell = self._build_shell_with_fm([_fake_fm('fm_alpha')])
        returned = shell.run_line_magic('ws3_inspect_model', '')

        assert returned is None
        assert len(displayed) == 1
        assert isinstance(displayed[0], Markdown)
        md_text = displayed[0].data
        assert '### Inspect Model' in md_text
        assert '`tsa24_clipped`' in md_text
        assert '`2020`' in md_text
        assert '`37.5`' in md_text

    def test_multiple_models_lists_candidates_not_calls_provider(self, monkeypatch):
        """Ambiguous queries must list every candidate and not invoke the provider."""
        from ws3.agent import ipython_magics as mod

        displayed = []
        monkeypatch.setattr(mod, 'display', displayed.append)
        monkeypatch.setattr(mod.InspectModel, 'run', lambda self_inst, *a, **kw: None)

        shell = self._build_shell_with_fm([
            _fake_fm('fm_alpha'),
            _fake_fm('fm_beta'),
        ])
        returned = shell.run_line_magic('ws3_inspect_model', '')

        assert returned is None
        assert len(displayed) == 1
        assert isinstance(displayed[0], Markdown)
        md_text = displayed[0].data
        assert 'fm_alpha' in md_text
        assert 'fm_beta' in md_text
        assert 'Specify which model' in md_text

    def test_single_model_match_by_variable_name(self, monkeypatch):
        """An explicit variable name resolves to exactly one candidate."""
        from ws3.agent import ipython_magics as mod

        displayed = []
        monkeypatch.setattr(mod, 'display', displayed.append)
        result = self._make_fake_result()
        monkeypatch.setattr(
            mod.InspectModel, 'run', lambda self_inst, *a, **kw: result
        )

        shell = self._build_shell_with_fm([
            _fake_fm('fm_alpha', model_name='alpha'),
            _fake_fm('fm_beta', model_name='beta'),
        ])
        returned = shell.run_line_magic('ws3_inspect_model', 'fm_beta')

        assert returned is None
        assert len(displayed) == 1
        assert isinstance(displayed[0], Markdown)
        md_text = displayed[0].data
        # The rendered output is the inspect result; the selection logic is
        # verified by the fact that the magic ran to completion without
        # listing candidates or raising an ambiguity error.
        assert '### Inspect Model' in md_text
        assert '`tsa24_clipped`' in md_text

    def test_no_model_returns_actionable_message(self, monkeypatch):
        """Without any ForestModel the magic displays a clear actionable message."""
        monkeypatch.setattr(ipython_magics, 'display', lambda obj: None)

        shell = InteractiveShell()
        load_ipython_extension(shell)
        displayed = []
        monkeypatch.setattr(ipython_magics, 'display', displayed.append)
        returned = shell.run_line_magic('ws3_inspect_model', '')

        assert returned is None
        assert len(displayed) == 1
        assert isinstance(displayed[0], Markdown)
        md_text = displayed[0].data
        assert 'No ForestModel found' in md_text
        assert 'fm = ForestModel(...)' in md_text

    def test_query_by_model_name_selects_exactly_one(self, monkeypatch):
        """When two candidates exist, a query mentioning one candidate's
        ``model_name`` (e.g. ``alpha``) resolves to exactly that candidate and
        renders the inspect result — no ambiguity list is shown."""
        from ws3.agent import ipython_magics as mod

        displayed = []
        monkeypatch.setattr(mod, 'display', displayed.append)
        result = self._make_fake_result()
        monkeypatch.setattr(
            mod.InspectModel, 'run', lambda self_inst, *a, **kw: result
        )

        shell = self._build_shell_with_fm([
            _fake_fm('fm_alpha', model_name='alpha', name='alpha'),
            _fake_fm('fm_beta', model_name='beta', name='beta'),
        ])
        returned = shell.run_line_magic('ws3_inspect_model', 'alpha')

        assert returned is None
        assert len(displayed) == 1
        assert isinstance(displayed[0], Markdown)
        md_text = displayed[0].data
        assert '### Inspect Model' in md_text
        assert '`tsa24_clipped`' in md_text
        assert 'ambiguous query' not in md_text
        assert 'Specify which model' not in md_text

    def test_query_by_public_name_selects_exactly_one(self, monkeypatch):
        """When two candidates exist, a query mentioning one candidate's
        public ``name`` (e.g. ``pine_model``) resolves to exactly that
        candidate and renders the inspect result — no ambiguity list is
        shown."""
        from ws3.agent import ipython_magics as mod

        displayed = []
        monkeypatch.setattr(mod, 'display', displayed.append)
        result = self._make_fake_result()
        monkeypatch.setattr(
            mod.InspectModel, 'run', lambda self_inst, *a, **kw: result
        )

        shell = self._build_shell_with_fm([
            _fake_fm('fm_pine', model_name='other', name='pine_model'),
            _fake_fm('fm_spruce', model_name='other', name='spruce_model'),
        ])
        returned = shell.run_line_magic('ws3_inspect_model', 'pine_model')

        assert returned is None
        assert len(displayed) == 1
        assert isinstance(displayed[0], Markdown)
        md_text = displayed[0].data
        assert '### Inspect Model' in md_text
        assert '`tsa24_clipped`' in md_text
        assert 'ambiguous query' not in md_text
        assert 'Specify which model' not in md_text


class TestSubclassDiscovery:
    """Regression: ``_find_models`` accepts ``ForestModel`` subclasses via ``isinstance``."""

    def test_find_models_accepts_subclass(self):
        """A ``ForestModel`` subclass instance is found alongside an exact-type fake."""
        from ws3.agent.ipython_magics import _find_models

        class MySubclass(ForestModel):
            pass

        sub = MySubclass.__new__(MySubclass)
        sub.model_name = 'sub'
        sub.periods = [1]
        sub.namespace_key = 'sub_model'

        exact = _FakeForestModel.__new__(_FakeForestModel)
        exact.model_name = 'exact'
        exact.periods = [1]
        exact.namespace_key = 'exact_model'

        shell = InteractiveShell()
        shell.user_ns['sub_model'] = sub
        shell.user_ns['exact_model'] = exact

        found = _find_models(shell)
        assert len(found) == 2
        names = {n for n, _ in found}
        assert 'sub_model' in names
        assert 'exact_model' in names

    def test_find_fm_returns_first_subclass(self):
        """``_find_fm`` also matches a subclass when it is the only candidate."""
        from ws3.agent.ipython_magics import _find_fm

        class MySubclass(ForestModel):
            pass

        sub = MySubclass.__new__(MySubclass)
        sub.model_name = 'only_one'

        shell = InteractiveShell()
        shell.user_ns['only_one'] = sub

        picked = _find_fm(shell)
        assert picked is sub


# ---------------------------------------------------------------------------
# Phase 8 — `_match_identifier` unit tests
# ---------------------------------------------------------------------------


class TestMatchIdentifier:
    """Verify that `_match_identifier` enforces complete-token matching."""

    def test_rejects_substring_query(self):
        """A bare ``fm`` must not match a longer identifier like ``fm_alpha``."""
        from ws3.agent.ipython_magics import _match_identifier
        assert _match_identifier('fm', 'fm_alpha') is False

    def test_rejects_partial_word_match(self):
        """``spruce`` must not match ``spruce_fir`` -- only full-token match."""
        from ws3.agent.ipython_magics import _match_identifier
        assert _match_identifier('spruce', 'spruce_fir') is False

    def test_accepts_complete_match(self):
        """The full identifier matches itself."""
        from ws3.agent.ipython_magics import _match_identifier
        assert _match_identifier('fm_alpha', 'fm_alpha') is True

    def test_accepts_case_insensitive(self):
        """Casing differences are ignored; the token set comparison is lowercased."""
        from ws3.agent.ipython_magics import _match_identifier
        assert _match_identifier('FM_ALPHA', 'fm_alpha') is True

    def test_accepts_superset_query(self):
        """A query containing extra context words still matches the candidate."""
        from ws3.agent.ipython_magics import _match_identifier
        assert _match_identifier('my fm_alpha model', 'fm_alpha') is True

    def test_blank_query_is_not_a_match(self):
        """An empty query short-circuits and never matches anything."""
        from ws3.agent.ipython_magics import _match_identifier
        assert _match_identifier('', 'fm_alpha') is False

    def test_blank_candidate_is_not_a_match(self):
        """An empty candidate string also short-circuits."""
        from ws3.agent.ipython_magics import _match_identifier
        assert _match_identifier('fm_alpha', '') is False

    def test_different_tokens_do_not_match(self):
        """Unrelated identifiers share no tokens and should not match."""
        from ws3.agent.ipython_magics import _match_identifier
        assert _match_identifier('alpha', 'beta') is False


# ---------------------------------------------------------------------------
# Phase 8 — `_select_model` unit tests
# ---------------------------------------------------------------------------


class TestSelectModel:
    """Verify the three resolution outcomes of `_select_model`."""

    def test_exact_single_resolution(self):
        """A query matching exactly one candidate returns that model and None reason."""
        from ws3.agent.ipython_magics import _select_model
        fm_a = _fake_fm('fm_a', model_name='alpha')
        fm_b = _fake_fm('fm_b', model_name='beta')

        picked, reason = _select_model([
            ('fm_a', fm_a), ('fm_b', fm_b)], 'alpha')

        assert picked is fm_a
        assert reason is None

    def test_ambiguous_matches(self):
        """If the query matches more than one candidate, return 'ambiguous'."""
        from ws3.agent.ipython_magics import _select_model
        fm_x = _fake_fm('fm_shared_x', model_name='shared')
        fm_y = _fake_fm('fm_shared_y', model_name='shared')

        _, reason = _select_model([
            ('fm_shared_x', fm_x), ('fm_shared_y', fm_y)], 'shared')

        assert reason == 'ambiguous'

    def test_no_match_returns_none(self):
        """When the query matches nothing, return None and 'no_match'."""
        from ws3.agent.ipython_magics import _select_model
        fm_a = _fake_fm('fm_a', model_name='alpha')

        picked, reason = _select_model([('fm_a', fm_a)], 'unicorn')

        assert picked is None
        assert reason == 'no_match'

    def test_empty_models_returns_no_match(self):
        """An empty model list yields no_match."""
        from ws3.agent.ipython_magics import _select_model
        picked, reason = _select_model([], 'anything')
        assert picked is None
        assert reason == 'no_match'


# ---------------------------------------------------------------------------
# Phase 8 — multi-model magic paths
# ---------------------------------------------------------------------------


class TestInspectModelMagicMultiModel:
    """Verify that the multi-model magic does NOT call InspectModel.run."""

    def test_no_match_lists_candidates_and_does_not_call_inspect_model(
        self, monkeypatch,
    ):
        """When no model matches the query, the magic lists candidates
        and returns without invoking InspectModel.run."""
        from ws3.agent import ipython_magics as mod

        displayed = []
        monkeypatch.setattr(mod, 'display', displayed.append)

        run_calls = []
        def _spy_run(self_inst, *args, **kwargs):
            run_calls.append(True)
            return None

        monkeypatch.setattr(mod.InspectModel, 'run', _spy_run)

        shell = InteractiveShell()
        load_ipython_extension(shell)
        shell.user_ns['fm_alpha'] = _fake_fm('fm_alpha')
        shell.user_ns['fm_beta'] = _fake_fm('fm_beta')

        # 'xyzzy' matches neither candidate, so reason == 'no_match'.
        returned = shell.run_line_magic('ws3_inspect_model', 'xyzzy')

        assert returned is None
        assert len(run_calls) == 0, 'InspectModel.run must not be called'
        assert len(displayed) == 1
        md_text = displayed[0].data
        assert 'fm_alpha' in md_text
        assert 'fm_beta' in md_text
        assert 'Specify which model' in md_text

    def test_ambiguous_query_does_not_call_inspect_model(
        self, monkeypatch,
    ):
        """When the query is ambiguous, the magic shows the ambiguity
        message and does NOT invoke InspectModel.run."""
        from ws3.agent import ipython_magics as mod

        displayed = []
        monkeypatch.setattr(mod, 'display', displayed.append)

        run_calls = []
        def _spy_run(self_inst, *args, **kwargs):
            run_calls.append(True)
            return None

        monkeypatch.setattr(mod.InspectModel, 'run', _spy_run)

        shell = InteractiveShell()
        load_ipython_extension(shell)
        # Two models sharing the same public name triggers ambiguity.
        shell.user_ns['fm_x'] = _fake_fm('fm_x', model_name='twins')
        shell.user_ns['fm_y'] = _fake_fm('fm_y', model_name='twins')

        returned = shell.run_line_magic('ws3_inspect_model', 'twins')

        assert returned is None
        assert len(run_calls) == 0, 'InspectModel.run must not be called'
        assert len(displayed) == 1
        md_text = displayed[0].data
        assert 'ambiguous query' in md_text


# ---------------------------------------------------------------------------
# Phase 8 — `_fmt_verdict` zero-value and None handling
# ---------------------------------------------------------------------------


class TestFmtVerdictZeroValues:
    """Verify `_fmt_verdict` renders zero-valued counts and total_area=0.0"""

    def test_zero_valued_counts_render_as_values(self):
        """All zero-valued integer counts must appear as their literal value
        rather than being skipped as falsy."""
        from ws3.agent.capabilities.inspect_model import InspectResult

        v = InspectResult(
            model_name='bare',
            name='bare',
            base_year=0,
            horizon=0,
            period_length=0.0,
            periods=[],
            nthemes=0,
            nactions=0,
            ndtypes=0,
            total_area=0.0,
        )
        result = SimpleNamespace(ok=True, value=v, errors=[])
        rendered = ipython_magics._fmt_verdict('inspect_model', result)

        assert '- **model_name**: `bare`' in rendered
        assert '- **base_year**: `0`' in rendered
        assert '- **horizon**: `0`' in rendered
        assert '- **period_length**: `0.0`' in rendered
        assert '- **periods**: `[]`' in rendered
        assert '- **nthemes**: `0`' in rendered
        assert '- **nactions**: `0`' in rendered
        assert '- **ndtypes**: `0`' in rendered

    def test_total_area_zero_is_rendered_not_unavailable(self):
        """``total_area=0.0`` must render as ``0.0``, not as 'unavailable'."""
        from ws3.agent.capabilities.inspect_model import InspectResult

        v = InspectResult(
            model_name='bare',
            name='bare',
            base_year=None,
            horizon=None,
            period_length=None,
            periods=None,
            nthemes=0,
            nactions=0,
            ndtypes=0,
            total_area=0.0,
        )
        result = SimpleNamespace(ok=True, value=v, errors=[])
        rendered = ipython_magics._fmt_verdict('inspect_model', result)

        assert '- **total_area (period 1)**: `0.0`' in rendered
        assert 'unavailable' not in rendered

    def test_total_area_none_stays_unavailable(self):
        """``total_area=None`` should continue to render as 'unavailable'."""
        from ws3.agent.capabilities.inspect_model import InspectResult

        v = InspectResult(
            model_name='bare',
            name='bare',
            base_year=None,
            horizon=None,
            period_length=None,
            periods=None,
            nthemes=0,
            nactions=0,
            ndtypes=0,
            total_area=None,
        )
        result = SimpleNamespace(ok=True, value=v, errors=[])
        rendered = ipython_magics._fmt_verdict('inspect_model', result)

        assert '- **total_area (period 1)**: unavailable' in rendered
