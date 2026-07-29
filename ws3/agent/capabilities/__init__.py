"""
ws3 capability registry.

Assembled lazily: capabilities are constructed on demand rather than at import
time, so that importing :py:mod:`ws3.agent` stays cheap and does not require a
model to exist.
"""

from __future__ import annotations

from typing import Any, Optional

from fresh_agent_core.registry import Registry

from ws3.agent.capabilities.build_mask import BuildMask, MaskRequest
from ws3.agent.capabilities.diagnose_import import DiagnoseImport, Diagnosis, ImportFailure
from ws3.agent.capabilities.explain_exception import (
    ExceptionReport,
    ExplainException,
    Explanation,
)

__all__ = [
    'build_registry',
    'BuildMask',
    'DiagnoseImport',
    'ExplainException',
    'MaskRequest',
    'ExceptionReport',
    'ImportFailure',
    'Explanation',
    'Diagnosis',
]


def build_registry(fm: Optional[Any] = None) -> Registry:
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
    ])
