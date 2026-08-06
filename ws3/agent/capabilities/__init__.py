"""
ws3 capability registry.

Assembled lazily: capabilities are constructed on demand rather than at import
time, so that importing :py:mod:`ws3.agent` stays cheap and does not require a
model to exist.
"""

from __future__ import annotations

from typing import Any

from fresh_agent_core.registry import Registry

from ws3.agent.capabilities.build_mask import BuildMask, BuildMaskOutput, MaskRequest
from ws3.agent.capabilities.diagnose_import import DiagnoseImport, Diagnosis, ImportFailure
from ws3.agent.capabilities.explain_exception import (
    ExceptionReport,
    ExplainException,
    Explanation,
)
from ws3.agent.capabilities.inspect_model import InspectInputs, InspectModel, InspectResult
from ws3.agent.capabilities.rtfm_capability import RTFMCapability, RTFMInputs, RTFMResult
from ws3.agent.capabilities.ws3_hint import HintInputs, HintResult, Ws3Hint

__all__ = [
    'build_registry',
    'BuildMask',
    'BuildMaskOutput',
    'DiagnoseImport',
    'ExplainException',
    'HintInputs',
    'HintResult',
    'InspectInputs',
    'InspectModel',
    'InspectResult',
    'MaskRequest',
    'ExceptionReport',
    'ImportFailure',
    'Explanation',
    'Diagnosis',
    'RTFMCapability',
    'RTFMInputs',
    'RTFMResult',
    'Ws3Hint',
]


def build_registry(fm: Any | None = None) -> Registry:
    """
    Build the ws3 capability registry.

    :param fm: Optional :py:class:`~ws3.forest.ForestModel`. When supplied,
        :py:class:`~ws3.agent.capabilities.build_mask.BuildMask` describes the
        model's real theme codes in its prompt, which turns most of the task from
        generation into selection. Without it the model must guess and the
        validator will usually reject the result.
    """
    return Registry([
        BuildMask(fm),
        ExplainException(),
        DiagnoseImport(),
        InspectModel(),
        RTFMCapability(),
        Ws3Hint(),
    ])
