"""
MCP server exposing ws3's capabilities.

Turns the capability surface from a convention an agent must remember into a tool
list it can see. Instructions get ignored; tools in the tool list get called.

Run it::

    ws3-agent-mcp --model-path examples/data/woodstock_model_files_tsa24_clipped \\
                  --model-name tsa24_clipped

The model path matters. ``build_mask`` validates proposals by resolving them
against a real :py:class:`~ws3.forest.ForestModel`, so without one it will
correctly -- but uselessly -- reject everything. Some context cannot travel over
the wire, which is why the server is constructed with access to it rather than
receiving it per call.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

__all__ = ['build_ws3_server', 'main']


def _load_model(model_path: str, model_name: str, **kwargs: Any) -> Any:
    """Load and partially import a ForestModel for use as validator context."""
    from ws3.forest import ForestModel

    fm = ForestModel(  # type: ignore[no-untyped-call]
        model_name=model_name,
        model_path=model_path,
        base_year=kwargs.get('base_year', 2020),
        horizon=kwargs.get('horizon', 10),
        period_length=kwargs.get('period_length', 10),
        max_age=kwargs.get('max_age', 1000),
    )
    # Landscape defines the themes build_mask validates against; areas populate
    # the development types a mask has to match. Later sections are not needed to
    # validate a mask and are skipped so startup stays fast.
    fm.import_landscape_section()  # type: ignore[no-untyped-call]
    fm.import_areas_section(convert_periods_to_years=kwargs.get('period_length', 10))  # type: ignore[no-untyped-call]
    return fm


def build_ws3_server(
    *,
    model_path: str | None = None,
    model_name: str | None = None,
    provider: Any = None,
    config: Any = None,
    sink: Any = None,
    **model_kwargs: Any,
) -> Any:
    """
    Build the ws3 MCP server.

    :param model_path: Directory holding Woodstock model files. Without it
        ``build_mask`` has nothing to validate against.
    :param model_name: Base name of the model files.
    :param provider: Override the model backend, e.g. a ``FakeProvider``.
    :param config: Override the resolved configuration.
    :param sink: Where provenance records go.
    :param model_kwargs: Passed to :py:class:`~ws3.forest.ForestModel`.
    """
    import fresh_agent_core as core
    from fresh_agent_core.mcp import build_server

    from ws3.agent.capabilities import build_registry
    from ws3.agent.capabilities.diagnose_import import ImportFailure

    fm = None
    if model_path and model_name:
        fm = _load_model(model_path, model_name, **model_kwargs)

    resolved = config if config is not None else core.config.resolve()
    if resolved is None:
        if provider is None:
            raise core.AgentUnavailable(
                'No agent configuration found. Set FRESH_AGENT_ENDPOINT and '
                'FRESH_AGENT_MODEL, or write ~/.config/fresh-agent/config.toml.'
            )
        resolved = core.AgentConfig(endpoint='offline://test', model='offline')

    if provider is None:
        provider = core.OpenAIProvider(resolved)

    def context_factory(name: str, arguments: dict[str, Any]) -> Any:
        """
        Supply each capability's validator context.

        build_mask needs the loaded model. diagnose_import needs the failure
        being diagnosed, which the caller does send, so it is reconstructed here
        rather than held on the server. inspect_model also needs the loaded
        model so the deterministic executor can read live metadata fields.
        """
        if name == 'build_mask':
            return fm
        if name == 'diagnose_import':
            return ImportFailure(
                model_path=arguments.get('model_path', model_path or ''),
                model_name=arguments.get('model_name', model_name or ''),
                section=arguments.get('section', ''),
                error=arguments.get('error', ''),
                excerpt=arguments.get('excerpt', ''),
            )
        if name == 'inspect_model':
            return fm
        return None

    return build_server(
        build_registry(fm),
        server_name='ws3',
        provider=provider,
        config=resolved,
        context_factory=context_factory,
        sink=sink,
    )


def main(argv: list[str] | None = None) -> int:
    """Console entry point for ``ws3-agent-mcp``."""
    parser = argparse.ArgumentParser(
        prog='ws3-agent-mcp',
        description='MCP server exposing ws3 capabilities, each validated against '
                    'real model state.',
    )
    parser.add_argument(
        '--model-path',
        help='Directory holding Woodstock model files. Without it, build_mask has '
             'nothing to validate proposals against and will reject everything.',
    )
    parser.add_argument('--model-name', help='Base name of the model files.')
    parser.add_argument('--base-year', type=int, default=2020)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--period-length', type=int, default=10)
    parser.add_argument('--max-age', type=int, default=1000)
    parser.add_argument(
        '--provenance-log',
        help='Path for the provenance JSONL. Defaults to '
             '$FRESH_AGENT_LOG, else ./.fresh-agent/provenance.jsonl',
    )
    parser.add_argument(
        '--list-tools',
        action='store_true',
        help='Print the tool descriptors and exit, without starting a server.',
    )
    args = parser.parse_args(argv)

    if (args.model_path is None) != (args.model_name is None):
        parser.error('--model-path and --model-name must be given together')

    if args.list_tools:
        import json

        from fresh_agent_core.mcp import describe_tools

        from ws3.agent.capabilities import build_registry

        print(json.dumps(describe_tools(build_registry()), indent=2))
        return 0

    from fresh_agent_core.provenance import JSONLSink

    server = build_ws3_server(
        model_path=args.model_path,
        model_name=args.model_name,
        sink=JSONLSink(args.provenance_log) if args.provenance_log else JSONLSink(),
        base_year=args.base_year,
        horizon=args.horizon,
        period_length=args.period_length,
        max_age=args.max_age,
    )

    import anyio
    from mcp.server.stdio import stdio_server

    async def _serve() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(_serve)
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
