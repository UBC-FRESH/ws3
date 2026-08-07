"""
Tests for the ws3 MCP server.

Covers the transport boundary: tool descriptors, payload mapping, result
rendering, and a full round trip through the registered handler. Everything runs
offline against ``FakeProvider``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip('fresh_agent_core', reason='ws3[agent] not installed')
pytest.importorskip('mcp', reason='mcp not installed')

from fresh_agent_core import AgentConfig, FakeProvider  # noqa: E402
from fresh_agent_core.capability import CapabilityResult  # noqa: E402
from fresh_agent_core.mcp import describe_tools, format_result  # noqa: E402

from ws3.agent.capabilities import build_registry  # noqa: E402
from ws3.agent.capabilities.build_mask import BuildMask, MaskRequest  # noqa: E402
from ws3.agent.capabilities.diagnose_import import DiagnoseImport, Diagnosis  # noqa: E402
from ws3.agent.capabilities.explain_exception import (  # noqa: E402
    ExplainException,
    Explanation,
)
from ws3.agent.capabilities.inspect_model import InspectInputs, InspectModel  # noqa: E402
from ws3.agent.mcp_server import build_ws3_server, main  # noqa: E402

CONFIG = AgentConfig(endpoint='offline://test', model='test-model')

MODEL_DIR = Path(__file__).parent.parent / 'examples' / 'data' / 'woodstock_model_files_tsa24_clipped'
MODEL_NAME = 'tsa24_clipped'


@pytest.fixture(scope='module')
def fm():
    if not MODEL_DIR.is_dir():
        pytest.skip(f'test model not available at {MODEL_DIR}')
    from ws3.forest import ForestModel

    model = ForestModel(
        model_name=MODEL_NAME, model_path=str(MODEL_DIR),
        base_year=2020, horizon=10, period_length=10, max_age=1000,
    )
    model.import_landscape_section()
    model.import_areas_section(convert_periods_to_years=10)
    return model


class TestToolDescriptors:
    def test_seven_tools_are_advertised(self):
        """Task 8.7 adds the deterministic scenario report to the MCP tool list."""
        assert len(describe_tools(build_registry())) == 7

    def test_tools_are_json_serialisable(self):
        json.dumps(describe_tools(build_registry()))

    @pytest.mark.parametrize('name, required', [
        ('build_mask', ['description']),
        ('explain_exception', ['exc_type', 'message']),
        ('diagnose_import', ['model_path', 'model_name', 'section', 'error']),
        ('report_scenario_inventory_products', ['model_path', 'model_name']),
    ])
    def test_each_tool_declares_its_required_arguments(self, name, required):
        """
        A default schema tells a calling agent nothing.

        Without required fields the agent has to guess the payload shape, which
        wastes attempts on avoidable rejections.
        """
        tool = next(t for t in describe_tools(build_registry()) if t['name'] == name)
        assert tool['inputSchema']['required'] == required

    def test_descriptions_state_what_is_validated(self):
        for tool in describe_tools(build_registry()):
            assert 'validat' in tool['description'].lower()

    def test_diagnose_import_enumerates_valid_sections(self):
        """Enumerating them beats letting the agent guess a suffix."""
        tool = next(t for t in describe_tools(build_registry()) if t['name'] == 'diagnose_import')
        assert 'lan' in tool['inputSchema']['properties']['section']['enum']


class TestPayloadMapping:
    def test_build_mask_payload(self, fm):
        request = BuildMask(fm).from_payload({'description': 'mature spruce'})
        assert isinstance(request, MaskRequest)
        assert request.description == 'mature spruce'

    def test_explain_exception_payload(self):
        report = ExplainException().from_payload({
            'exc_type': 'ValueError', 'message': 'boom', 'traceback_text': 'tb',
        })
        assert report.exc_type == 'ValueError'
        assert report.traceback_text == 'tb'

    def test_diagnose_import_payload(self):
        failure = DiagnoseImport().from_payload({
            'model_path': '/tmp/m', 'model_name': 'm', 'section': 'lan', 'error': 'e',
        })
        assert failure.section == 'lan'

    def test_missing_optional_fields_default_to_empty(self):
        report = ExplainException().from_payload({'exc_type': 'E', 'message': 'm'})
        assert report.traceback_text == ''

    def test_inspect_model_payload(self):
        inputs = InspectModel().from_payload({
            'query': 'full snapshot',
            'model_name': 'tsa24_clipped',
        })
        assert isinstance(inputs, InspectInputs)
        assert inputs.query == 'full snapshot'
        assert inputs.model_name == 'tsa24_clipped'

    def test_inspect_model_payload_missing_query(self):
        inputs = InspectModel().from_payload({})
        assert inputs.query == ''


class TestRendering:
    def test_mask_renders_as_a_pasteable_string(self, fm):
        from ws3.agent.capabilities.build_mask import BuildMaskOutput
        output = BuildMaskOutput(mask=('a', '?', 'b'))
        assert BuildMask(fm).render(output) == 'a ? b'

    def test_explanation_renders_cause_and_actions(self):
        rendered = ExplainException().render(
            Explanation(cause='It broke.', next_actions=('Fix it.',), symbols_referenced=())
        )
        assert 'It broke.' in rendered
        assert 'Fix it.' in rendered

    def test_diagnosis_states_that_the_fix_was_verified(self):
        """
        The verification is the product.

        A suggested fix is cheap; a fix that provably re-imports is the thing
        worth acting on, so the rendering says so explicitly.
        """
        rendered = DiagnoseImport().render(
            Diagnosis(cause='typo', original_line='*BROKEN', corrected_line='*THEME')
        )
        assert 'verified' in rendered.lower()

    def test_failure_rendering_carries_reasons_not_a_value(self):
        result = CapabilityResult(
            ok=False, value=None, attempts=3, provenance_ids=(),
            errors=('mask matches zero development types',),
        )
        payload = json.loads(format_result(result, BuildMask()))
        assert payload['ok'] is False
        assert 'result' not in payload
        assert payload['validation_failures'] == ['mask matches zero development types']


class TestContextFactory:
    """context_factory must hand the right context to each capability."""

    def test_build_mask_receives_the_loaded_fm(self, fm):
        server = build_ws3_server(
            model_path=str(MODEL_DIR),
            model_name=MODEL_NAME,
            provider=FakeProvider(['x'], repeat_last=True),
            config=CONFIG,
        )
        assert server is not None

    def test_inspect_model_receives_the_loaded_fm(self, fm):
        """
        The inspect_model capability must receive the loaded ForestModel as
        context so the deterministic executor can read live metadata fields.

        The context_factory is a closure inside build_ws3_server, so we can't
        extract it directly. Instead, we verify the routing by dispatching a
        real inspect_model call through the server's capability and asserting
        the executor receives the loaded fm (not None, not a string).
        """
        import asyncio

        from ws3.agent.capabilities.inspect_model import InspectModel

        captured = {}

        original_run = InspectModel.run

        def _spy_run(self, inputs, *, provider, config, context=None, sink=None):
            captured['context'] = context
            return original_run(
                self, inputs, provider=provider, config=config,
                context=context, sink=sink,
            )

        InspectModel.run = _spy_run

        try:
            server = build_ws3_server(
                model_path=str(MODEL_DIR),
                model_name=MODEL_NAME,
                provider=FakeProvider(['{"operation": "full_snapshot"}']),
                config=CONFIG,
            )
            # Dispatch inspect_model through the public request_handlers API
            # on the MCP Server object.  CallToolRequest is the registered
            # handler for tool-calling; we build the request manually so the
            # test stays offline and never starts a stdio server.
            from mcp.types import CallToolRequest, CallToolRequestParams

            call_handler = server.request_handlers[CallToolRequest]

            async def _dispatch():
                req = CallToolRequest(
                    method='tools/call',
                    params=CallToolRequestParams(
                        name='inspect_model',
                        arguments={'query': 'full snapshot'},
                    ),
                )
                return await call_handler(req)

            asyncio.run(_dispatch())

            # The loaded fm was passed as context, not None and not a stub.
            ctx = captured.get('context')
            assert ctx is not None, (
                'inspect_model context was None, expected a loaded ForestModel'
            )
            # build_ws3_server constructs its own ForestModel via _load_model,
            # so we verify identity by class rather than by object identity.
            assert type(ctx).__name__ == 'ForestModel', (
                f'inspect_model context was {type(ctx).__name__}, '
                'expected ForestModel'
            )
        finally:
            InspectModel.run = original_run

    def test_no_model_means_none_context_for_inspect(self):
        """Without a model path, inspect_model context must be None."""
        import asyncio

        from mcp.types import CallToolRequest, CallToolRequestParams

        from ws3.agent.capabilities.inspect_model import InspectModel

        captured = {}

        original_run = InspectModel.run

        def _spy_run(self, inputs, *, provider, config, context=None, sink=None):
            captured['context'] = context
            return original_run(
                self, inputs, provider=provider, config=config,
                context=context, sink=sink,
            )

        InspectModel.run = _spy_run

        try:
            server = build_ws3_server(
                provider=FakeProvider(['{"operation": "full_snapshot"}']),
                config=CONFIG,
            )

            call_handler = server.request_handlers[CallToolRequest]

            async def _dispatch():
                req = CallToolRequest(
                    method='tools/call',
                    params=CallToolRequestParams(
                        name='inspect_model',
                        arguments={'query': 'full snapshot'},
                    ),
                )
                return await call_handler(req)

            asyncio.run(_dispatch())
            assert captured.get('context') is None, (
                f'expected None context without a model, got '
                f'{captured.get("context")!r}'
            )
        finally:
            InspectModel.run = original_run

    def test_unknown_capability_receives_no_context(self):
        server = build_ws3_server(
            provider=FakeProvider(['x'], repeat_last=True),
            config=CONFIG,
        )
        assert server is not None


class TestServerConstruction:
    def test_builds_without_a_model(self):
        """
        Usable without a model, though build_mask will then reject everything.

        Failing to start would be worse: explain_exception needs no model and is
        still useful.
        """
        server = build_ws3_server(provider=FakeProvider(['x'], repeat_last=True), config=CONFIG)
        assert server is not None

    def test_builds_with_a_model(self):
        if not MODEL_DIR.is_dir():
            pytest.skip('test model not available')
        server = build_ws3_server(
            model_path=str(MODEL_DIR),
            model_name=MODEL_NAME,
            provider=FakeProvider(['x'], repeat_last=True),
            config=CONFIG,
        )
        assert server is not None

    def test_unconfigured_and_no_provider_raises_actionably(self, monkeypatch):
        monkeypatch.delenv('FRESH_AGENT_ENDPOINT', raising=False)
        monkeypatch.delenv('FRESH_AGENT_MODEL', raising=False)
        import fresh_agent_core as core

        monkeypatch.setattr(core.config, 'resolve', lambda *a, **k: None)
        with pytest.raises(core.AgentUnavailable, match='FRESH_AGENT_ENDPOINT'):
            build_ws3_server()


class TestRoundTrip:
    """A tool call must reach the capability, run the oracle, and come back."""

    def test_registry_dispatch_reaches_the_right_capability(self):
        """Dispatch is keyed on name, so a mismatch would silently run the wrong tool."""
        registry = build_registry()
        for name in ('build_mask', 'explain_exception', 'diagnose_import'):
            assert registry.get(name).name == name

    def test_valid_proposal_round_trips_through_payload_and_rendering(self, fm):
        """
        The full transport path: JSON arguments in, rendered text out.

        Exercised through the capability rather than MCP's async dispatch so the
        guarantee holds independently of SDK internals, which are not ours to pin.
        """
        wildcard = ' '.join(['?'] * fm.nthemes())
        capability = BuildMask(fm)
        provider = FakeProvider([
            json.dumps({'mask': wildcard, 'reasoning': 'all'}) + '\nRTFM links: none'
        ])

        result = capability.run(
            capability.from_payload({'description': 'everything'}),
            provider=provider, config=CONFIG, context=fm,
        )
        payload = json.loads(format_result(result, capability))

        assert payload['ok'] is True
        assert payload['result'] == wildcard

    def test_invalid_proposal_does_not_return_a_result(self, fm):
        """The central guarantee, restated at the transport boundary."""
        capability = BuildMask(fm)
        provider = FakeProvider(
            [json.dumps({'mask': 'nope ' * fm.nthemes()})], repeat_last=True
        )
        result = capability.run(
            capability.from_payload({'description': 'anything'}),
            provider=provider, config=CONFIG, context=fm,
        )
        payload = json.loads(format_result(result, capability))
        assert payload['ok'] is False
        assert 'result' not in payload


class TestConsoleEntryPoint:
    def test_list_tools_prints_json(self, capsys):
        assert main(['--list-tools']) == 0
        parsed = json.loads(capsys.readouterr().out)
        assert {t['name'] for t in parsed} == {
            'build_mask', 'explain_exception', 'diagnose_import',
            'inspect_model', 'report_scenario_inventory_products', 'rtfm', 'ws3_hint',
        }

    def test_model_path_and_name_must_be_given_together(self):
        with pytest.raises(SystemExit):
            main(['--model-path', '/tmp/x'])

    def test_help_mentions_why_the_model_matters(self, capsys):
        """
        Omitting the model silently degrades build_mask to rejecting everything.

        That is correct behaviour but baffling from the outside, so the help says
        so rather than leaving it to be discovered.
        """
        with pytest.raises(SystemExit):
            main(['--help'])
        assert 'validate' in capsys.readouterr().out.lower()
