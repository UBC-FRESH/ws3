import sys
sys.path.append('../ws3/')
import textwrap
from pathlib import Path

import pytest
from ws3.forest import Action, DevelopmentType, ForestModel, _search


class TestLandscapeThemeDescriptions:
    """
    Theme descriptions must survive import.

    Theme order and meaning are entirely user-defined in the Woodstock format, so
    the text trailing a ``*THEME`` declaration is the only thing in the dataset
    that states what a theme position represents. The importer used to discard it,
    which left every theme anonymous downstream.
    """

    @pytest.fixture
    def model(self, tmp_path: Path) -> ForestModel:
        (tmp_path / 'm.lan').write_text(textwrap.dedent("""\
            *THEME Timber Supply Area (TSA)
            tsa24
            *THEME Leading tree species
            sw
            pl
            *AGGREGATE conifer
            sw pl
            *THEME
            1
            2
            """))
        fm = ForestModel(model_name='m', model_path=str(tmp_path),
                         base_year=2020, horizon=1, period_length=10, max_age=100)
        fm.import_landscape_section()
        return fm

    def test_descriptions_are_extracted(self, model):
        assert model._themes[0]['__description__'] == 'Timber Supply Area (TSA)'
        assert model._themes[1]['__description__'] == 'Leading tree species'

    def test_undescribed_theme_yields_empty_description(self, model):
        """Absent is absent -- it must not be filled in with the placeholder name."""
        assert model._themes[2]['__description__'] == ''

    def test_theme_count_and_codes_are_unaffected(self, model):
        assert model.nthemes() == 3
        assert model.theme_basecodes(0) == ['tsa24']
        assert model.theme_basecodes(1) == ['sw', 'pl']
        assert model.theme_basecodes(2) == ['1', '2']

    def test_aggregates_still_parse(self, model):
        """The declaration line now feeds a capture group; aggregates must survive."""
        assert model._themes[1]['conifer'] == ['sw', 'pl']


def test_search_returns_match_when_pattern_matches():
    """_search behaves like re.search on the happy path."""
    m = _search(r'(?<=\().*(?=\))', 'MULTIPLY(a, b)', 'test construct')
    assert m.group(0) == 'a, b'


def test_search_raises_descriptive_error_when_pattern_does_not_match():
    """
    Malformed parser input must produce a useful message.

    The Woodstock parsers previously called ``re.search(...).group(...)`` directly.
    On malformed input the search returns None and the chained ``.group()`` raised
    ``AttributeError: 'NoneType' object has no attribute 'group'``, naming neither
    the construct being parsed nor the offending text.
    """
    with pytest.raises(ValueError) as exc:
        _search(r'(?<=\().*(?=\))', 'MULTIPLY no parens here', 'MULTIPLY arguments')
    msg = str(exc.value)
    assert 'MULTIPLY arguments' in msg
    assert 'expected pattern' in msg
    assert 'MULTIPLY no parens here' in msg


def test_search_error_is_not_attribute_error():
    """Specifically guards against regressing to the old bare AttributeError."""
    with pytest.raises(ValueError):
        _search(r'zzz', 'nothing matching', 'some construct')


class _StubParent:
    """Minimal stand-in for ForestModel exposing only what _compile_oper_expr reads."""
    horizon = 10
    max_age = 200


def _bare_dtype(acode='some_action'):
    """
    Build a DevelopmentType without running __init__.

    _compile_oper_expr only reads self.parent.horizon, self._max_age, and writes
    into self.operability, so a fully constructed ForestModel (which requires
    on-disk Woodstock model files) is unnecessary here and would make this a slow
    integration test rather than a focused regression test.
    """
    dt = object.__new__(DevelopmentType)
    dt.parent = _StubParent()
    dt._max_age = _StubParent.max_age
    dt.operability = {acode: {}}
    return dt


@pytest.mark.parametrize('expr, expected_periods', [
    ('_cp = 5', [5]),
    ('_cp >= 5', [5, 6, 7, 8, 9, 10]),
    ('_cp <= 5', [1, 2, 3, 4, 5]),
])
def test_compile_oper_expr_period_operators(expr, expected_periods):
    """
    Regression test for the '_cp' relational operator branches.

    Two defects were present here:

    1. The '<=' branch referenced an undefined name `rel_opertors` (missing 'a'),
       raising NameError.
    2. The bound was then folded in with an unguarded
       `plo, phi = max(_plo, plo), min(_phi, phi)`. A one-sided comparison leaves
       the opposite bound as None, so both '>=' and '<=' raised TypeError. The
       parallel '_age' branch guards for None; this one did not.

    Net effect: only '_cp =' worked. This test pins all three operators.
    """
    acode = 'some_action'
    dt = _bare_dtype(acode)
    dt._compile_oper_expr(acode, expr)
    assert sorted(dt.operability[acode].keys()) == expected_periods


def test_compile_oper_expr_rejects_bad_period_operator():
    """A relational operator outside {=, >=, <=} must raise ValueError."""
    acode = 'some_action'
    dt = _bare_dtype(acode)
    with pytest.raises(ValueError, match='Bad relational operator'):
        dt._compile_oper_expr(acode, '_cp != 5')


@pytest.mark.skip(
    reason="Requires an 'area_selector' fixture that is not defined anywhere. "
           "This function was also decorated with @pytest.fixture, which made pytest "
           "collect it as a fixture rather than a test, so it silently never ran. "
           "Unskip once a GreedyAreaSelector fixture with a populated ForestModel exists."
)
def test_operate(area_selector):
    period = 10
    acode = "some_action_code"
    target_area = 1000.0  # assuming a specific target area
    mask = None  # or set a mask if required
    commit_actions = True
    verbose = False

    remaining_area = area_selector.operate(period, acode, target_area, mask, commit_actions, verbose)

    assert isinstance(remaining_area, float)  # for example, asserting the type of the return value


def test_action_initialization():
    code = "some_code"
    targetage = None
    descr = ''
    lockexempt = False
    components = [] #need to be checked
    partial = [] #need to be checked
    is_harvest = 0
    is_sticky = 0

    action = Action(code, targetage, descr, lockexempt, components, partial, is_harvest, is_sticky)

    assert action.code == code
    assert action.targetage == targetage
    assert action.descr == descr
    assert action.lockexempt == lockexempt
    assert action.components == components
    assert action.partial == partial
    assert action.is_harvest == is_harvest
    assert action.is_sticky == is_sticky
    assert action.oper_a is None
    assert action.oper_p is None
    assert not action.is_compiled
    assert action.treatment_type is None
