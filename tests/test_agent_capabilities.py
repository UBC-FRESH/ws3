"""
Tests for the ws3 agent capabilities.

The point of these is not that the plumbing works -- ``fresh-agent-core`` already
proves that. It is that **the validators are real oracles**: they consult actual
ws3 state and reject output that is well-formed, plausible, and wrong.

Everything runs offline against ``FakeProvider``. No endpoint, no credentials.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip('fresh_agent_core', reason='ws3[agent] not installed')

from fresh_agent_core import AgentConfig, FakeProvider  # noqa: E402
from fresh_agent_core.provenance import MemorySink  # noqa: E402

import ws3.agent  # noqa: E402
from ws3.agent.capabilities.build_mask import BuildMask, MaskRequest  # noqa: E402
from ws3.agent.capabilities.diagnose_import import (  # noqa: E402
    DiagnoseImport,
    Diagnosis,
    ImportFailure,
)
from ws3.agent.capabilities.explain_exception import (  # noqa: E402
    ExceptionReport,
    ExplainException,
    Explanation,
    extract_symbols,
    known_symbols,
)

CONFIG = AgentConfig(endpoint='offline://test', model='test-model')

MODEL_DIR = Path(__file__).parent.parent / 'examples' / 'data' / 'woodstock_model_files_tsa24_clipped'
MODEL_NAME = 'tsa24_clipped'


@pytest.fixture(scope='module')
def fm():
    """A real ForestModel. The validators are only meaningful against real state."""
    if not MODEL_DIR.is_dir():
        pytest.skip(f'test model not available at {MODEL_DIR}')
    from ws3.forest import ForestModel

    model = ForestModel(
        model_name=MODEL_NAME,
        model_path=str(MODEL_DIR),
        base_year=2020,
        horizon=10,
        period_length=10,
        max_age=1000,
    )
    model.import_landscape_section()
    model.import_areas_section(convert_periods_to_years=10)
    return model


def _mask_response(mask: str) -> str:
    return json.dumps({'mask': mask, 'reasoning': 'because'})


def _explanation_response(cause: str, actions: list[str]) -> str:
    return json.dumps({'cause': cause, 'next_actions': actions})


class TestAgentPackage:
    def test_importing_ws3_does_not_import_the_agent_package(self):
        """
        `import ws3` must not pull in the optional agent stack.

        Otherwise core modelling would inherit an optional dependency, and in a
        deployment without it `import ws3` would fail outright.
        """
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, '-c', 'import sys, ws3; print("ws3.agent" in sys.modules)'],
            capture_output=True, text=True, check=True,
        )
        assert result.stdout.strip() == 'False'

    def test_available_is_false_without_configuration(self, monkeypatch):
        monkeypatch.delenv('FRESH_AGENT_ENDPOINT', raising=False)
        monkeypatch.delenv('FRESH_AGENT_MODEL', raising=False)
        assert ws3.agent.available() is False

    def test_available_never_raises(self):
        assert isinstance(ws3.agent.available(), bool)

    def test_three_capabilities_are_registered(self):
        assert len(ws3.agent.list_capabilities()) == 3

    def test_descriptions_say_what_is_validated(self):
        """
        The description is what an agent reads to decide what guarantee it gets.

        If it does not state the validation, the caller cannot tell a validated
        capability from an unchecked one.
        """
        for capability in ws3.agent.list_capabilities():
            assert 'validat' in capability['description'].lower()

    def test_unknown_capability_lists_the_real_ones(self):
        with pytest.raises(KeyError, match='build_mask'):
            ws3.agent.get('no_such_capability')


class TestBuildMaskValidator:
    """The oracle: does the proposed mask resolve to at least one dtype?"""

    def test_wildcard_mask_is_valid(self, fm):
        mask = tuple(['?'] * fm.nthemes())
        assert BuildMask(fm).validate(mask, fm).ok is True

    def test_mask_matching_a_real_dtype_is_valid(self, fm):
        key = list(fm.dtypes)[0]
        assert BuildMask(fm).validate(tuple(key), fm).ok is True

    def test_mask_matching_nothing_is_rejected(self, fm):
        """
        The case that matters.

        Syntactically fine, right length, entirely useless. Only resolving it
        against the model reveals that.
        """
        mask = tuple(['nonexistent-code'] * fm.nthemes())
        verdict = BuildMask(fm).validate(mask, fm)
        assert verdict.ok is False
        assert any('zero development types' in e for e in verdict.errors)

    def test_rejection_names_the_offending_codes(self, fm):
        """A retry needs an actionable reason, not just 'no'."""
        mask = ('definitely-not-a-code',) + tuple(['?'] * (fm.nthemes() - 1))
        verdict = BuildMask(fm).validate(mask, fm)
        assert any('definitely-not-a-code' in e for e in verdict.errors)

    def test_wrong_length_mask_is_rejected(self, fm):
        verdict = BuildMask(fm).validate(('?',), fm)
        assert verdict.ok is False
        assert any('entries' in e for e in verdict.errors)

    def test_missing_context_is_reported_as_a_caller_error(self, fm):
        verdict = BuildMask(fm).validate(('?',), None)
        assert verdict.ok is False
        assert any('caller error' in e for e in verdict.errors)


class TestBuildMaskParsing:
    def test_parses_a_space_separated_string(self, fm):
        assert BuildMask(fm).parse(_mask_response('a b c')) == ('a', 'b', 'c')

    def test_parses_a_list(self, fm):
        assert BuildMask(fm).parse(json.dumps({'mask': ['a', 'b']})) == ('a', 'b')

    def test_lowercases(self, fm):
        assert BuildMask(fm).parse(_mask_response('THEME1 Theme2')) == ('theme1', 'theme2')

    def test_tolerates_fenced_json(self, fm):
        """
        Models wrap JSON in code fences despite instructions.

        Tolerating it is cheaper than spending a retry on formatting.
        """
        raw = '```json\n' + _mask_response('a b') + '\n```'
        assert BuildMask(fm).parse(raw) == ('a', 'b')

    def test_rejects_non_json(self, fm):
        from fresh_agent_core.capability import ParseError
        with pytest.raises(ParseError, match='mask'):
            BuildMask(fm).parse('I think the mask should be ? ? ?')

    def test_rejects_json_without_a_recognised_key(self, fm):
        from fresh_agent_core.capability import ParseError
        with pytest.raises(ParseError, match='constraints'):
            BuildMask(fm).parse(json.dumps({'answer': 'a b'}))


class TestBuildMaskArityIsStructural:
    """
    Mask arity comes from the model, never from the language model.

    The theme count is authoritative state read from the LANDSCAPE section, so a
    capability that asks the model to reproduce it is inventing a failure mode.
    A live 9B model returned ten entries for this five-theme model, which is what
    prompted assembling the mask here instead.
    """

    def _constraints(self, mapping):
        return json.dumps({'constraints': mapping, 'reasoning': 'because'})

    def test_no_constraints_yields_all_wildcards(self, fm):
        mask = BuildMask(fm).parse(self._constraints({}))
        assert mask == tuple(['?'] * fm.nthemes())

    def test_sparse_constraints_are_padded_to_full_arity(self, fm):
        key = list(fm.dtypes)[0]
        mask = BuildMask(fm).parse(self._constraints({'1': key[1]}))
        assert len(mask) == fm.nthemes()
        assert mask[1] == key[1].lower()
        assert all(m == '?' for i, m in enumerate(mask) if i != 1)

    def test_arity_holds_for_every_subset_of_constrained_themes(self, fm):
        """The property that makes the redesign worth it, checked exhaustively."""
        capability = BuildMask(fm)
        key = list(fm.dtypes)[0]
        for size in range(fm.nthemes() + 1):
            mapping = {str(i): key[i] for i in range(size)}
            mask = capability.parse(self._constraints(mapping))
            assert len(mask) == fm.nthemes(), f'arity broke for {size} constraints'

    def test_assembled_mask_always_resolves(self, fm):
        """Assembly plus the real oracle: a wildcard-padded key must match."""
        capability = BuildMask(fm)
        key = list(fm.dtypes)[0]
        mask = capability.parse(self._constraints({'0': key[0]}))
        assert capability.validate(mask, fm).ok is True

    def test_out_of_range_position_is_rejected(self, fm):
        from fresh_agent_core.capability import ParseError
        with pytest.raises(ParseError, match='out of range'):
            BuildMask(fm).parse(self._constraints({str(fm.nthemes()): 'x'}))

    def test_non_integer_position_is_rejected(self, fm):
        from fresh_agent_core.capability import ParseError
        with pytest.raises(ParseError, match='not a position number'):
            BuildMask(fm).parse(self._constraints({'not_a_theme': 'x'}))

    def test_theme_name_is_accepted_as_a_position(self, fm):
        """
        A live model keys constraints by the theme name it was shown.

        The name identifies the theme unambiguously, so resolving it is
        deterministic. Rejecting it burned retries and, worse, pushed the model
        into declaring a perfectly groundable request ungroundable.
        """
        key = list(fm.dtypes)[0]
        name = fm._themes[1]['__name__']
        mask = BuildMask(fm).parse(self._constraints({name: key[1]}))
        assert len(mask) == fm.nthemes()
        assert mask[1] == key[1].lower()

    def test_theme_name_matching_is_case_insensitive(self, fm):
        key = list(fm.dtypes)[0]
        name = fm._themes[1]['__name__'].upper()
        assert BuildMask(fm).parse(self._constraints({name: key[1]}))[1] == key[1].lower()

    def test_constraints_must_be_an_object(self, fm):
        from fresh_agent_core.capability import ParseError
        with pytest.raises(ParseError, match='must be a mapping'):
            BuildMask(fm).parse(json.dumps({'constraints': ['a', 'b']}))

    def test_prompt_does_not_ask_the_model_to_assemble_the_mask(self, fm):
        from ws3.agent.capabilities.build_mask import MaskRequest
        prompt = BuildMask(fm).build_messages(MaskRequest('everything'), ())[0]['content']
        assert 'Do not assemble the mask yourself' in prompt
        assert str(fm.nthemes()) in prompt


class TestBuildMaskAggregates:
    """
    Aggregates are valid mask values and carry the modeller's own semantics.

    ``unmask`` expands them via ``_expand_theme``, so treating one as an unknown
    code would reject a better answer than the model could otherwise give.
    """

    @pytest.fixture
    def fm_agg(self):
        from ws3.forest import ForestModel

        model = ForestModel(
            model_name='agg', model_path='.', base_year=2020,
            horizon=1, period_length=10, max_age=100,
        )
        model.add_theme('species', basecodes=['sw', 'pl', 'aw'],
                        aggs={'conifer': ['sw', 'pl']})
        model.add_theme('site', basecodes=['good', 'poor'])
        return model

    def test_prompt_lists_aggregates_separately_from_basecodes(self, fm_agg):
        from ws3.agent.themes import ThemeSchema
        summary = ThemeSchema.from_model(fm_agg).describe()
        assert 'aggregate conifer: sw, pl' in summary
        # The aggregate must not be muddled in with the basecode listing.
        codes_line = next(l for l in summary.splitlines() if l.strip().startswith('codes:'))
        assert 'conifer' not in codes_line

    def test_aggregate_is_accepted_as_a_code(self, fm_agg):
        capability = BuildMask(fm_agg)
        mask = capability.parse(
            json.dumps({'constraints': {'0': 'conifer'}, 'reasoning': 'x'})
        )
        assert mask == ('conifer', '?')
        # No dtypes are loaded, so this cannot resolve -- but the failure must be
        # "matched nothing", not "conifer is not a code for this theme".
        assert not any(
            'not a code' in e for e in capability.validate(mask, fm_agg).errors
        )

    def test_unknown_code_feedback_offers_the_aggregate(self, fm_agg):
        verdict = BuildMask(fm_agg).validate(('softwood', '?'), fm_agg)
        assert verdict.ok is False
        assert any('conifer' in e for e in verdict.errors)


class TestRunSuppliesModelAtConstruction:
    """
    ``ws3.agent.run`` must build the capability *with* the model.

    Every other test here constructs ``BuildMask(fm)`` by hand, so none of them
    exercised the real entry point -- which built a model-blind capability and
    could not assemble a mask at all. Caught only by a live call; kept as a
    regression test because the offline suite was structurally blind to it.
    """

    def test_run_can_assemble_a_mask(self, fm):
        raw = json.dumps({'constraints': {}, 'reasoning': 'everything'})
        result = ws3.agent.run(
            'build_mask', 'all stands',
            context=fm, provider=FakeProvider([raw], repeat_last=True),
        )
        assert result.ok is True, result.errors
        assert result.value == tuple(['?'] * fm.nthemes())

    def test_get_passes_the_model_through(self, fm):
        from ws3.agent.capabilities.build_mask import MaskRequest
        capability = ws3.agent.get('build_mask', fm)
        prompt = capability.build_messages(MaskRequest('x'), ())[0]['content']
        assert f'has {fm.nthemes()} themes' in prompt

    def test_non_model_context_is_not_mistaken_for_a_model(self, fm):
        """
        Other capabilities take other context, which must not reach BuildMask.

        The guard sits in ``run``, where ``context`` is deliberately untyped;
        ``get`` takes an explicit model by contract.
        """
        from ws3.agent import _as_forest_model
        assert _as_forest_model(fm) is fm
        assert _as_forest_model(None) is None
        assert _as_forest_model(ImportFailure(
            model_path='p', model_name='n', section='lan', error='e',
        )) is None

    def test_run_tolerates_a_non_model_context(self):
        """Constructing the registry must not explode on unrelated context."""
        raw = json.dumps({
            'cause': 'something went wrong', 'next_actions': ['check the file'],
            'symbols_referenced': [],
        })
        result = ws3.agent.run(
            'explain_exception', ValueError('boom'),
            context=None, provider=FakeProvider([raw], repeat_last=True),
        )
        assert result.ok is True, result.errors


class TestBuildMaskEndToEnd:
    def test_invalid_mask_never_reaches_the_caller(self, fm):
        capability = BuildMask(fm)
        provider = FakeProvider([_mask_response('bogus ' * fm.nthemes())], repeat_last=True)

        result = capability.run(
            MaskRequest('anything'), provider=provider, config=CONFIG, context=fm
        )

        assert result.ok is False
        assert result.value is None

    def test_recovers_after_an_invalid_proposal(self, fm):
        capability = BuildMask(fm)
        wildcard = ' '.join(['?'] * fm.nthemes())
        provider = FakeProvider([
            _mask_response('nope ' * fm.nthemes()),
            _mask_response(wildcard),
        ])

        result = capability.run(
            MaskRequest('everything'), provider=provider, config=CONFIG, context=fm
        )

        assert result.ok is True
        assert result.attempts == 2

    def test_rejection_reason_is_fed_back_into_the_retry(self, fm):
        capability = BuildMask(fm)
        wildcard = ' '.join(['?'] * fm.nthemes())
        provider = FakeProvider([
            _mask_response('badcode ' * fm.nthemes()),
            _mask_response(wildcard),
        ])
        capability.run(MaskRequest('x'), provider=provider, config=CONFIG, context=fm)

        assert 'rejected' in provider.calls[1][0]['content'].lower()

    def test_prompt_lists_real_theme_codes(self, fm):
        """
        Turns most of the task from generation into selection.

        Without the real codes the model invents plausible ones, which the
        validator correctly rejects -- burning the budget on avoidable failures.
        """
        capability = BuildMask(fm)
        messages = capability.build_messages(MaskRequest('x'), ())
        assert MODEL_NAME in messages[0]['content']

    def test_provenance_records_every_attempt(self, fm):
        sink = MemorySink()
        capability = BuildMask(fm)
        provider = FakeProvider([_mask_response('bad ' * fm.nthemes())], repeat_last=True)

        capability.run(
            MaskRequest('x'), provider=provider, config=CONFIG, context=fm, sink=sink
        )

        assert sink.attempts == capability.max_attempts
        assert all(r.ok is False for r in sink.records)


class TestSymbolExtraction:
    def test_finds_dotted_ws3_references(self):
        found = extract_symbols('Call ws3.opt.Problem.solve to solve it.')
        assert 'ws3.opt.Problem.solve' in found

    def test_finds_bare_class_method_references(self):
        assert 'Problem.solve' in extract_symbols('Use Problem.solve here.')

    @pytest.mark.parametrize('text', [
        'Check os.path for details',
        'See e.g. the docs',
        'Use np.array instead',
        'self.parent is the model',
    ])
    def test_ignores_stdlib_and_prose(self, text):
        """Otherwise ordinary English produces false rejections and the loop thrashes."""
        assert extract_symbols(text) == []

    def test_known_symbols_includes_real_ws3_names(self):
        known = known_symbols()
        assert 'ForestModel' in known
        assert 'Problem' in known
        assert 'Curve' in known

    def test_known_symbols_includes_real_methods(self):
        assert 'unmask' in known_symbols()

    def test_known_symbols_excludes_names_that_were_never_written(self):
        """
        The defect class this capability exists to catch.

        Every one of these was referenced somewhere in ws3 -- docs, tests, or
        module code -- despite never having been implemented.
        """
        known = known_symbols()
        assert 'compile_scenario' not in known
        assert 'interpolate_curves' not in known
        assert 'get_objective_value' not in known


class TestExplainExceptionValidator:
    """The oracle: does every referenced ws3 name actually exist?"""

    def _explanation(self, cause, actions=('do something',)):
        capability = ExplainException()
        return capability.parse(_explanation_response(cause, list(actions)))

    def test_explanation_citing_real_symbols_is_valid(self):
        candidate = self._explanation(
            'The ForestModel could not resolve the mask.',
            ['Call unmask to check which development types match.'],
        )
        assert ExplainException().validate(candidate, None).ok is True

    def test_explanation_citing_a_fabricated_symbol_is_rejected(self):
        """
        The headline case.

        Fluent, helpful, and pointing at a method that does not exist. Exactly
        what Phase 6, 7.5 and 7.6 each had to remove by hand.
        """
        candidate = self._explanation(
            'The problem was not compiled.',
            ['Call ws3.core.compile_scenario to build the problem.'],
        )
        verdict = ExplainException().validate(candidate, None)
        assert verdict.ok is False
        assert any('compile_scenario' in e for e in verdict.errors)

    @pytest.mark.parametrize('fabricated', [
        'ws3.core.interpolate_curves',
        'Problem.get_objective_value',
        'ForestModel.simulate_everything',
    ])
    def test_known_fabrications_are_all_rejected(self, fabricated):
        candidate = self._explanation('Something failed.', [f'Call {fabricated}.'])
        assert ExplainException().validate(candidate, None).ok is False

    def test_empty_cause_is_rejected(self):
        candidate = Explanation(cause='  ', next_actions=('x',), symbols_referenced=())
        assert ExplainException().validate(candidate, None).ok is False

    def test_no_next_actions_is_rejected(self):
        candidate = Explanation(cause='Something failed.', next_actions=(), symbols_referenced=())
        assert ExplainException().validate(candidate, None).ok is False


class TestExplainExceptionEndToEnd:
    def test_fabricated_api_never_reaches_the_caller(self):
        capability = ExplainException()
        provider = FakeProvider([
            _explanation_response('Not compiled.', ['Call ws3.core.compile_scenario.']),
        ], repeat_last=True)

        result = capability.run(
            ExceptionReport('ValueError', 'boom'), provider=provider, config=CONFIG
        )

        assert result.ok is False
        assert result.value is None

    def test_recovers_when_a_later_attempt_avoids_fabrication(self):
        capability = ExplainException()
        provider = FakeProvider([
            _explanation_response('Not compiled.', ['Call ws3.core.compile_scenario.']),
            _explanation_response('The mask matched nothing.', ['Check unmask output.']),
        ])

        result = capability.run(
            ExceptionReport('ValueError', 'boom'), provider=provider, config=CONFIG
        )

        assert result.ok is True
        assert 'mask' in result.value.cause

    def test_the_offending_symbol_is_named_in_the_retry(self):
        capability = ExplainException()
        provider = FakeProvider([
            _explanation_response('x', ['Call ws3.core.compile_scenario.']),
            _explanation_response('The mask matched nothing.', ['Check unmask output.']),
        ])
        capability.run(ExceptionReport('ValueError', 'boom'), provider=provider, config=CONFIG)

        assert 'compile_scenario' in provider.calls[1][0]['content']


class TestDiagnoseImportValidator:
    """The strongest oracle: apply the fix and re-parse."""

    @pytest.fixture
    def broken_model(self, tmp_path):
        """
        A copy of the real model whose landscape section genuinely fails to parse.

        Every ``*THEME`` marker is replaced, not just the first. An earlier version
        of this fixture corrupted only the first one and the model still imported
        cleanly -- there are five declarations, and ``*THEMEX`` still matches the
        parser's ``\\*THEME.*`` pattern regardless. The fixture proved nothing and
        the tests built on it were vacuous, which is worth recording given how
        easily it looked correct.

        Verified below: the fixture asserts the model really is broken before any
        test uses it.
        """
        if not MODEL_DIR.is_dir():
            pytest.skip('test model not available')
        scratch = tmp_path / MODEL_DIR.name
        shutil.copytree(MODEL_DIR, scratch)
        lan = scratch / f'{MODEL_NAME}.lan'
        lan.write_text(lan.read_text().replace('*THEME', '*BROKEN'))
        return scratch

    def test_the_fixture_is_actually_broken(self, broken_model):
        """
        Guard the guard.

        If this passes silently, every other test in this class is meaningless.
        """
        from ws3.forest import ForestModel

        fm = ForestModel(
            model_name=MODEL_NAME, model_path=str(broken_model), base_year=2020
        )
        with pytest.raises(Exception):
            fm.import_landscape_section()

    def test_a_fix_that_works_is_accepted(self, broken_model):
        failure = ImportFailure(
            model_path=str(broken_model),
            model_name=MODEL_NAME,
            section='lan',
            error='ValueError: Could not parse landscape section',
        )
        candidate = Diagnosis(
            cause='THEME keyword corrupted',
            original_line='*BROKEN',
            corrected_line='*THEME',
        )
        assert DiagnoseImport().validate(candidate, failure).ok is True

    def test_a_fix_that_does_not_work_is_rejected(self, broken_model):
        """
        Plausible, well-formed, and does not actually fix anything.

        Only re-parsing reveals that, which is the whole argument for this oracle.
        """
        failure = ImportFailure(
            model_path=str(broken_model),
            model_name=MODEL_NAME,
            section='lan',
            error='ValueError: Could not parse landscape section',
        )
        candidate = Diagnosis(
            cause='wrong guess',
            original_line='*BROKEN',
            corrected_line='*STILLBROKEN',
        )
        verdict = DiagnoseImport().validate(candidate, failure)
        assert verdict.ok is False
        assert any('still fails' in e for e in verdict.errors)

    def test_a_no_op_fix_is_rejected(self, broken_model):
        failure = ImportFailure(
            model_path=str(broken_model), model_name=MODEL_NAME,
            section='lan', error='x',
        )
        candidate = Diagnosis(cause='x', original_line='*BROKEN', corrected_line='*BROKEN')
        verdict = DiagnoseImport().validate(candidate, failure)
        assert any('changes nothing' in e for e in verdict.errors)

    def test_a_line_that_is_not_present_is_rejected(self, broken_model):
        failure = ImportFailure(
            model_path=str(broken_model), model_name=MODEL_NAME,
            section='lan', error='x',
        )
        candidate = Diagnosis(
            cause='x',
            original_line='THIS LINE IS NOT IN THE FILE',
            corrected_line='something',
        )
        verdict = DiagnoseImport().validate(candidate, failure)
        assert any('not found' in e for e in verdict.errors)

    def test_unknown_section_is_rejected(self, broken_model):
        failure = ImportFailure(
            model_path=str(broken_model), model_name=MODEL_NAME,
            section='zzz', error='x',
        )
        candidate = Diagnosis(cause='x', original_line='a', corrected_line='b')
        assert DiagnoseImport().validate(candidate, failure).ok is False

    def test_missing_model_path_is_rejected(self):
        failure = ImportFailure(
            model_path='/no/such/path', model_name=MODEL_NAME, section='lan', error='x',
        )
        candidate = Diagnosis(cause='x', original_line='a', corrected_line='b')
        verdict = DiagnoseImport().validate(candidate, failure)
        assert any('does not exist' in e for e in verdict.errors)

    def test_validation_does_not_mutate_the_original_model(self, broken_model):
        """
        Capabilities are advisory. The fix is tested on a scratch copy.

        If validation edited the caller's files, a rejected suggestion would leave
        the model worse than it found it.
        """
        lan = broken_model / f'{MODEL_NAME}.lan'
        before = lan.read_text()

        failure = ImportFailure(
            model_path=str(broken_model), model_name=MODEL_NAME, section='lan', error='x',
        )
        DiagnoseImport().validate(
            Diagnosis(cause='x', original_line='*BROKEN', corrected_line='*THEME'),
            failure,
        )

        assert lan.read_text() == before
