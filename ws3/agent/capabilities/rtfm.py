"""
Shared RTFM footer utilities for ws3 agent capabilities.

Every capability response ends with a "RTFM links" section citing every module
path, function, class, and URL the response referenced. This is the librarian
walking alongside every capability call.

Design in: planning/phase8_embedded_agents.md — "Cross-cutting concern: RTFM
footer on every response".
"""

from __future__ import annotations

import importlib
import json
import re
import urllib.error
import urllib.request

from fresh_agent_core.capability import ParseError, Verdict

#: Modules whose public surface counts as "real ws3".
WS3_MODULES = (
    'ws3.common',
    'ws3.core',
    'ws3.forest',
    'ws3.opt',
    'ws3.spatial',
    'ws3.financial',
)

#: Matches dotted references like ``ws3.opt.Problem.solve`` or ``Problem.solve``.
_SYMBOL_PATTERN = re.compile(r'\b((?:ws3\.)?[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)\b')

#: Roots that look like dotted references but are not ws3 API.
_IGNORE_ROOTS = frozenset({
    'self', 'cls', 'os', 'sys', 'np', 'numpy', 'pd', 'pandas', 'json', 're',
    'math', 'pathlib', 'typing', 'dict', 'list', 'str', 'int', 'float', 'bool',
    'set', 'tuple', 'e', 'i', 'g', 'etc', 'vs',
})

#: File extension suffixes that appear in URLs but are not ws3 symbol segments.
_URL_EXTENSIONS = frozenset({
    'html', 'htm', 'py', 'rst', 'md', 'txt', 'json', 'xml', 'css', 'js',
    'png', 'jpg', 'svg', 'pdf',
})

#: Canonical doc base URL.
_DOCS_BASE = 'https://ubc-fresh.github.io/ws3'

#: The instruction appended to every capability prompt (unless include_rtfm=False).
RTFM_FOOTER_INSTRUCTION = """
After your response, append a "RTFM links" section citing every ws3 module path,
function, class, and doc URL you referenced in your answer. Use these canonical
URL forms:
  - Module docs: https://ubc-fresh.github.io/ws3/<module>.html
  - Symbol docs: https://ubc-fresh.github.io/ws3/<module>.html#<symbol>
  - Section docs: https://ubc-fresh.github.io/ws3/<section>/

If you referenced no ws3 symbols, write "RTFM links: none" and list zero links.
"""


def extract_json(text: str) -> tuple[str, str]:
    """
    Extract a JSON object from model output that may contain reasoning text.

    Handles:
    - ``<thinking>...</thinking>`` blocks before the JSON
    - Markdown fenced code blocks (`````json ... `````) wrapping the JSON
    - Free-text reasoning before or after the JSON object
    - Trailing text after the closing ``}`` (e.g. "RTFM links: ...")

    Returns a tuple of ``(json_text, footer_text)`` where ``footer_text`` is
    everything after the JSON object (typically the RTFM footer).

    Raises :py:class:`ParseError` if no valid JSON object can be found.

    Uses balanced brace counting to find the true JSON boundary, rather than
    a fragile last-``}`` heuristic that breaks when the model emits ``}`` in
    reasoning text.
    """
    # Strip <thinking>...</thinking> blocks the model may emit before the JSON.
    cleaned = re.sub(r'<thinking>.*?</thinking>\s*', '', text, flags=re.DOTALL)

    # Strip markdown fences if present.
    if cleaned.startswith('```'):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.DOTALL)

    # Find the JSON object using balanced brace counting.
    start = cleaned.find('{')
    if start == -1:
        raise ParseError(
            f'expected a JSON object, got: {text[:200]!r}'
        )

    depth = 0
    json_end = -1
    in_string = False
    escape_next = False

    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                json_end = i
                break

    if json_end == -1:
        raise ParseError(
            f'expected a JSON object, got: {text[:200]!r}'
        )

    json_text = cleaned[start:json_end + 1]

    # Validate it parses as JSON.
    try:
        json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ParseError(
            f'expected a JSON object, got invalid JSON: {exc.msg} at line '
            f'{exc.lineno}:{exc.colno}'
        ) from exc

    # Everything after the JSON object is the footer.
    footer = cleaned[json_end + 1:].strip()
    # Strip trailing fences if the model put them there.
    footer = re.sub(r'^```[^\n]*\n', '', footer).strip('` \n')

    return json_text, footer


def _extract_symbols(text: str) -> list[str]:
    """Pull dotted references that look like ws3 API mentions out of *text*."""
    found = []
    for match in _SYMBOL_PATTERN.finditer(text):
        symbol = match.group(1)
        root = symbol.split('.')[0].lower()
        if root in _IGNORE_ROOTS:
            continue
        # Filter out URL-like references (contain file extensions like .html)
        parts = symbol.split('.')
        if any(part.lower() in _URL_EXTENSIONS for part in parts):
            continue
        found.append(symbol)
    return found


def _public_names(module_name: str) -> set[str]:
    """Public attribute names of a module, plus its classes' public methods."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return set()
    names: set[str] = set()
    for attr in dir(module):
        if attr.startswith('_'):
            continue
        names.add(attr)
        value = getattr(module, attr, None)
        if isinstance(value, type):
            names.update(m for m in dir(value) if not m.startswith('_'))
    return names


def _known_symbols(modules: tuple[str, ...] = WS3_MODULES) -> set[str]:
    """Every public name across the given ws3 modules."""
    names: set[str] = set()
    for module_name in modules:
        names.update(_public_names(module_name))
        names.add(module_name.split('.')[-1])
    return names


def _doc_url_valid(url: str) -> bool:
    """Return True if a doc URL returns HTTP 200."""
    if not url.startswith(_DOCS_BASE):
        return False
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'ws3-agent/1.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return bool(resp.status == 200)
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def validate_rtfm_footer(
    response_text: str,
    *,
    footer_text: str | None = None,
    include_rtfm: bool = True,
) -> Verdict:
    """
    Validate the RTFM footer on a capability response.

    :param response_text: The full raw response text from the model.
    :param footer_text: Pre-extracted footer text (the content after "RTFM links:").
        When supplied, *response_text* is only used to detect the presence/absence
        of the marker when ``include_rtfm`` is False.
    :param include_rtfm: If False, the footer must be absent. If True, the footer
        must be present with valid citations.
    :return: ``Verdict.valid()`` if the footer is correct, ``Verdict.invalid()``
        with specific reasons if not.
    """
    rtfm_marker = 'RTFM links:'

    # --- include_rtfm=False: footer must be completely absent ---
    if not include_rtfm:
        marker_pos = response_text.rfind(rtfm_marker)
        if marker_pos != -1:
            return Verdict.invalid(
                'RTFM footer present but include_rtfm=False was set; '
                'suppress the RTFM links section'
            )
        return Verdict.valid()

    # --- include_rtfm=True: footer must be present with valid content ---
    if footer_text is not None:
        if response_text and rtfm_marker not in response_text:
            return Verdict.invalid(
                'RTFM footer missing; every response must end with a "RTFM links:" '
                'section citing every ws3 symbol and doc URL referenced'
            )
        # Pre-extracted footer (via parse()) follows a confirmed marker.
        footer = footer_text
    else:
        # Full-response path: find the footer marker at the end of the text
        marker_pos = response_text.rfind(rtfm_marker)
        if marker_pos == -1:
            return Verdict.invalid(
                'RTFM footer missing; every response must end with a "RTFM links:" '
                'section citing every ws3 symbol and doc URL referenced'
            )
        footer = response_text[marker_pos:]

    # "RTFM links: none" is valid (cited nothing)
    if footer.startswith(rtfm_marker + ' none'):
        return Verdict.valid()

    # Extract cited symbols and URLs from the footer
    cited_symbols = _extract_symbols(footer)
    cited_urls = re.findall(r'https://ubc-fresh\.github\.io/ws3/[^\s\)]+', footer)

    if not cited_symbols and not cited_urls:
        # Footer present but cites nothing — acceptable if model explicitly says "none"
        stripped = footer[len(rtfm_marker):].strip()
        if stripped and stripped.lower() != 'none':
            return Verdict.invalid(
                'RTFM footer present but appears empty; '
                'write "RTFM links: none" if no symbols were referenced'
            )
        return Verdict.valid()

    # Check every cited symbol exists in ws3
    known = _known_symbols()
    fabricated = []
    for symbol in cited_symbols:
        parts = symbol.split('.')
        leaf = parts[-1]
        touches_ws3 = any(part in known for part in parts[:-1])
        if touches_ws3 and leaf not in known:
            fabricated.append(symbol)

    if fabricated:
        return Verdict.invalid(
            'RTFM footer cites ws3 symbols that do not exist: '
            + ', '.join(sorted(set(fabricated)))
        )

    # Check every cited URL returns 200
    bad_urls = [url for url in cited_urls if not _doc_url_valid(url)]
    if bad_urls:
        return Verdict.invalid(
            'RTFM footer cites doc URLs that return errors: '
            + ', '.join(sorted(set(bad_urls)))
        )

    return Verdict.valid()
