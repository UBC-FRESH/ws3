"""
Theme structure of a :py:class:`~ws3.forest.ForestModel`, for agent capabilities.

The Woodstock model format is deliberately open ended: the user declares their own
set of themes, and their own set of stratification variable codes within each
theme. Nothing about that vocabulary is known ahead of time, and much of ws3's
internal complexity follows from it.

Two consequences drive this module:

**Theme count is structural.** The number of themes fixes the length of every
development-type key, and therefore of every mask, in a given model instance. It
is authoritative state, read from the LANDSCAPE section of the input dataset (or
from whatever built the model). A capability that asks a language model to
reproduce it has invented a failure mode: a live 9B model duly returned a
ten-entry mask for a five-theme model. Masks are assembled here, from sparse
constraints, so the arity is correct by construction rather than by policing.

**The code vocabulary belongs to the user.** No fixed set of codes is ever valid
across models, so prompts must carry the instance's actual vocabulary and
validation must resolve against the instance. Aggregates matter especially: an
aggregate is a name the user chose for a group of codes, so it is usually the
closest thing in the model to the words a person will use when describing the
stands they mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

#: Cap on codes listed per theme in a prompt. Real models can carry hundreds;
#: listing all of them buries the task and wastes context.
MAX_CODES_PER_THEME = 40

#: Cap on members listed when expanding an aggregate in a prompt.
MAX_AGGREGATE_MEMBERS = 12

#: Mask entry matching any code in a theme position.
WILDCARD = '?'


class ThemeError(ValueError):
    """A theme position or code that does not exist in this model instance."""


@dataclass(frozen=True)
class Theme:
    """
    One theme of a model instance.

    :param index: Position in the development-type key.
    :param name: Theme name, e.g. ``theme0``.
    :param description: Theme description, often empty.
    :param basecodes: Codes the user declared directly.
    :param aggregates: Aggregate code to the codes it stands for.
    """

    index: int
    name: str
    description: str
    basecodes: tuple[str, ...]
    aggregates: dict[str, tuple[str, ...]]

    def known(self, code: str) -> bool:
        """Whether *code* is a basecode or an aggregate of this theme."""
        return code in self.basecodes or code in self.aggregates

    def codes(self) -> tuple[str, ...]:
        """Every code valid in this theme position, basecodes and aggregates."""
        return self.basecodes + tuple(self.aggregates)


class ThemeSchema:
    """
    The theme structure of one model instance.

    Built from a :py:class:`~ws3.forest.ForestModel`; carries no reference to it,
    so it is safe to hold, describe and diff.
    """

    def __init__(self, themes: tuple[Theme, ...]) -> None:
        self.themes = themes

    @classmethod
    def from_model(cls, fm: Any) -> 'ThemeSchema':
        """
        Read the theme structure out of a model.

        Basecodes and aggregates are distinguished by the shape of the stored
        value -- ws3 stores a basecode as ``{code: code}`` and an aggregate as
        ``{code: [members]}`` -- rather than via
        :py:meth:`~ws3.forest.ForestModel.theme_basecodes`, because that parallel
        list is only appended to when a theme is declared with a non-empty
        basecode list, so it can fall out of step with ``_themes``.
        """
        themes = []
        for index, raw in enumerate(fm._themes):
            basecodes = []
            aggregates = {}
            for code, value in raw.items():
                if code.startswith('__'):
                    continue
                if isinstance(value, (list, tuple, set)):
                    aggregates[code] = tuple(value)
                else:
                    basecodes.append(code)
            themes.append(
                Theme(
                    index=index,
                    name=raw.get('__name__', f'theme{index}'),
                    description=raw.get('__description__', ''),
                    basecodes=tuple(basecodes),
                    aggregates=aggregates,
                )
            )
        return cls(tuple(themes))

    def __len__(self) -> int:
        return len(self.themes)

    @property
    def nthemes(self) -> int:
        """Number of themes, and so the length of every development-type key."""
        return len(self.themes)

    def wildcard_mask(self) -> tuple[str, ...]:
        """A mask matching every development type in the model."""
        return tuple([WILDCARD] * self.nthemes)

    def assemble(self, constraints: dict[Any, Any]) -> tuple[str, ...]:
        """
        Expand sparse theme constraints into a full-length mask.

        Positions not mentioned become wildcards, so the result always has exactly
        one entry per theme regardless of what was supplied. This is what keeps
        mask arity out of a language model's hands.

        :param constraints: Theme position to code. Keys may be ``int`` or the
            string form of one.
        :raises ThemeError: If a position is not an integer or is out of range.
        """
        if not isinstance(constraints, dict):
            raise ThemeError(
                f'constraints must be a mapping of theme position to code, got '
                f'{type(constraints).__name__}'
            )

        mask = list(self.wildcard_mask())
        for key, code in constraints.items():
            mask[self._position(key)] = str(code).lower()
        return tuple(mask)

    def _position(self, key: Any) -> int:
        """
        Resolve a constraint key to a valid theme index.

        Accepts the position number or the theme's name. The listing shown to a
        model puts both in front of it, and a model that keys by the name is not
        wrong about the theme -- only about the notation. Names are unambiguous
        here, so resolving them is deterministic rather than a guess, and it costs
        nothing compared with a retry.
        """
        if isinstance(key, str):
            name = key.strip().lower()
            for theme in self.themes:
                if theme.name.lower() == name:
                    return theme.index

        try:
            index = int(key)
        except (TypeError, ValueError):
            names = ', '.join(t.name for t in self.themes)
            raise ThemeError(
                f'theme position {key!r} is not a position number or a known theme '
                f'name; use one of the position numbers 0 to {self.nthemes - 1}, '
                f'or one of: {names}'
            ) from None
        if not 0 <= index < self.nthemes:
            raise ThemeError(
                f'theme position {index} is out of range; this model has '
                f'{self.nthemes} themes, numbered 0 to {self.nthemes - 1}'
            )
        return index

    def unknown_codes(self, mask: tuple[str, ...]) -> list[str]:
        """
        Describe mask entries that are not codes of their theme.

        Turns "matched nothing" into something a retry can act on, and names what
        would have worked -- the vocabulary is user-defined, so the model cannot
        infer it.
        """
        problems = []
        for index, code in enumerate(mask):
            if code == WILDCARD or index >= self.nthemes:
                continue
            theme = self.themes[index]
            if theme.known(code):
                continue
            valid = list(theme.codes())
            shown = ', '.join(valid[:MAX_CODES_PER_THEME]) or '(none)'
            suffix = '' if len(valid) <= MAX_CODES_PER_THEME else ' ...'
            problems.append(
                f'position {index} ({theme.name}): {code!r} is not a code for this '
                f'theme (valid: {shown}{suffix})'
            )
        return problems

    def describe(self) -> str:
        """
        Render the theme vocabulary for a prompt.

        Aggregates are listed separately and expanded, because an aggregate is a
        name the user chose for a group of stands and is therefore the code most
        likely to match how a person describes them.

        A theme with no description is marked as such. ``import_landscape_section``
        names themes ``theme0``, ``theme1`` and so on; those are positional
        placeholders carrying no meaning, and saying so is the difference between
        selecting on evidence and pattern-matching on a number.
        """
        lines = []
        for theme in self.themes:
            header = f'  position {theme.index} ({theme.name})'
            if theme.description:
                header += f' -- {theme.description}'
            else:
                header += ' -- undescribed, no stated meaning'
            lines.append(header + ':')

            shown = list(theme.basecodes[:MAX_CODES_PER_THEME])
            suffix = (
                '' if len(theme.basecodes) <= MAX_CODES_PER_THEME
                else f' ... ({len(theme.basecodes)} total)'
            )
            lines.append(f'      codes: {", ".join(shown) or "(none)"}{suffix}')

            for name, members in theme.aggregates.items():
                listed = ', '.join(members[:MAX_AGGREGATE_MEMBERS])
                more = (
                    '' if len(members) <= MAX_AGGREGATE_MEMBERS
                    else f' ... ({len(members)} total)'
                )
                lines.append(f'      aggregate {name}: {listed}{more}')
        return '\n'.join(lines)


def schema_for(fm: Optional[Any]) -> Optional[ThemeSchema]:
    """Return the schema for *fm*, or ``None`` if no model was supplied."""
    return None if fm is None else ThemeSchema.from_model(fm)
