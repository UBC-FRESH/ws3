import sys
sys.path.append('../ws3/')
import pytest
from ws3.forest import GreedyAreaSelector, Action, DevelopmentType

@pytest.fixture

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
    assert action.is_compiled == False
    assert action.treatment_type is None



