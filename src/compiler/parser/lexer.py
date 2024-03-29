import string
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, InitVar
from typing import NewType

from ...utils import aenumerate

KEYWORDS = {
    'and',
    'as',
    'assert',
    'break',
    'catch',
    'continue',
    'defer',
    'else',
    'enum',
    'exception',
    'false',
    'flags',
    'for',
    'from',
    'func',
    'if',
    'import',
    'in',
    'interface',
    'is',
    'iter',
    'let',
    'match',
    'module',
    'mutable',
    'nand',
    'none',
    'nor',
    'not',
    'or',
    'self',
    'then',
    'throw',
    'true',
    'try',
    'type',
    'var',
    'xnor',
    'xor',
    'yield',
}
OPERATORS = {
    '+': 'add',
    '-': 'sub',
    '*': 'mult',
    '/': 'div',
    '%': 'mod',
    '**': 'pow',
    '!': 'inv',
    '=': 'eq',
    '!=': 'neq',
    '<': 'lt',
    '<=': 'le',
    '>': 'gt',
    '>=': 'ge',
    '|': 'or',
    '!|': 'nor',
    '^': 'xor',
    '!^': 'xnor',
    '&': 'and',
    '!&': 'nand',
    '<<': 'lshift',
    '>>': 'rshift',
    '->': 'arrow',
    '..': 'range',
    '?': 'maybe',
    '+=': 'addeq',
    '-=': 'subeq',
    '*=': 'multeq',
    '/=': 'diveq',
    '%=': 'modeq',
    '**=': 'poweq',
    '|=': 'oreq',
    '!|=': 'noreq',
    '^=': 'xoreq',
    '!^=': 'xnoreq',
    '&=': 'andeq',
    '!&=': 'nandeq',
    '<<=': 'lshifteq',
    '>>=': 'rshifteq',
}
DELIMITERS = {
    '.': 'dot',
    ',': 'comma',
    ':': 'colon',
    '(': 'lbracket',
    ')': 'rbracket',
    '[': 'lsquare',
    ']': 'rsquare',
    '{': 'lbrace',
    '}': 'rbrace',
    '//': 'line_comment',
    '/*': 'block_comment',
}
INTEGER_TYPES = {
    'b': (set('01'),             'BINARY'),
    'o': (set(string.octdigits), 'OCTAL'),
    'x': (set(string.hexdigits), 'HEXADECIMAL'),
}
QUOTES = set('\'"')
PUNCTUATION = set(token[0] for token in OPERATORS | DELIMITERS)
NEWLINES = set('\r\n')
WHITESPACE = set(string.whitespace) - NEWLINES

Position = NewType('Position', tuple[int, int])

@dataclass
class Token:
    kind: str
    value: str
    start: Position = field(compare=False, default=(-1, -1))
    end: Position = field(compare=False, default=(-1, -1))

    def __post_init__(self) -> None:
        if self.end == (-1, -1) and self.start != (-1, -1):
            self.end = self.start

    def __bool__(self) -> bool:
        return bool(self.kind)


NULL = Token('', '')

# Exceptions
class InvalidCharacter(Exception):
    def __init__(self, char: str, linenum: int, column: int) -> None:
        super().__init__(f'{char!r} @ {linenum}:{column}')


# Helper class
@dataclass
class Chars:
    lines: InitVar[AsyncIterator[str]]
    _iterator: AsyncIterator[tuple[str, Position]] = field(init=False)
    next: str = field(init=False, default='')
    position: Position = field(init=False, default=Position((-1, -1)))

    def __post_init__(self, lines: AsyncIterator[str]) -> None:
        self._iterator = self._iterate(lines)

    @staticmethod
    async def new(lines: AsyncIterator[str]) -> 'Chars':
        self = Chars(lines)
        await self.advance()
        return self

    @staticmethod
    async def _iterate(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str, Position]]:
        linenum, column = 1, 0
        async for linenum, line in aenumerate(lines, start=linenum):
            column = 0
            for column, char in enumerate(line, start=column):
                yield char, Position((linenum, column))
        while True:
            yield '', Position((linenum, column + 1))

    async def advance(self) -> str:
        next = self.next
        self.next, self.position = await self._iterator.__anext__()
        return next


# Lexing functions
async def lex(lines: AsyncIterator[str]) -> AsyncIterator[Token]:
    chars = await Chars.new(lines)
    while True:
        char = chars.next
        start = chars.position
        if not char:
            kind, value = 'EOF', 'eof'
        elif char in NEWLINES:
            kind, value = await lex_newline(chars)
        elif char in WHITESPACE:
            await lex_whitespace(chars)
            continue
        elif char in string.digits:
            kind, value = await lex_number(chars)
        elif char in string.ascii_letters or char == '_':
            kind, value = await lex_name(chars)
        elif char in QUOTES:
            kind, value = await lex_string(chars)
        elif char in PUNCTUATION:
            kind, value = await lex_punctuation(chars)
        else:
            raise InvalidCharacter(char, *start)

        yield Token(kind, value, start, chars.position)


async def lex_newline(chars: Chars) -> tuple[str, str]:
    char = await chars.advance()
    if char == '\n':
        return 'NEWLINE', '\n'
    else:
        char = chars.next
        if char == '\n':
            await chars.advance()
            return 'NEWLINE', '\r\n'
        else:
            return 'NEWLINE', '\r'


async def lex_whitespace(chars: Chars) -> tuple[str, str]:
    value = ''
    while chars.next in WHITESPACE:
        value += chars.next
        await chars.advance()
    return 'WHITESPACE', value


async def lex_number(chars: Chars) -> tuple[str, str]:
    value = await chars.advance()
    if value == '0' and chars.next in INTEGER_TYPES:
        prefix = await chars.advance()
        value += prefix
        alphabet, kind = INTEGER_TYPES[prefix]
        value += await digits(chars, alphabet)
    else:
        kind = 'INTEGER'
        value += await digits(chars)
        if chars.next == '.':
            kind = 'NUMBER'
            value += await chars.advance()
            value += await digits(chars)
        if chars.next in ('e', 'E'):
            kind = 'NUMBER'
            value += await chars.advance()
            if chars.next in ('+', '-'):
                value += await chars.advance()
            value += await digits(chars)
        if chars.next in ('j', 'J'):
            value += await chars.advance()
            kind = f'IMAGINARY_{kind}'
    return kind, value


async def digits(chars: Chars, alphabet: set = set(string.digits)) -> str:
    value = ''
    while True:
        if chars.next == '_':
            value += await chars.advance()
            if chars.next not in alphabet:
                raise InvalidCharacter(chars.next, *chars.position)
        if chars.next in alphabet:
            value += await chars.advance()
        else:
            break

    return value


async def lex_name(chars: Chars) -> tuple[str, str]:
    value = ''
    alphabet = set(string.ascii_letters + string.digits + '_')
    while chars.next in alphabet:
        value += await chars.advance()

    if value in KEYWORDS:
        type = f'KW_{value.upper()}'
    else:
        type = 'NAME'
    return type, value


async def lex_string(chars: Chars) -> tuple[str, str]:
    value = quote = await chars.advance()
    while True:
        char = chars.next
        if not char or char in NEWLINES:
            raise InvalidCharacter(char or 'eof', *chars.position)
        elif char == quote:
            value += await chars.advance()
            break
        elif char == '\\':
            value += (await chars.advance()) + (await chars.advance())
        else:
            value += await chars.advance()

    return 'STRING', value


async def lex_punctuation(chars: Chars) -> tuple[str, str]:
    value = ''
    while chars.next and any(token.startswith(value + chars.next) for token in OPERATORS | DELIMITERS):
        value += await chars.advance()

    if value in OPERATORS:
        type = f'OP_{OPERATORS[value].upper()}'
    elif value in DELIMITERS:
        type = DELIMITERS[value].upper()
    else:
        raise InvalidCharacter(chars.next, *chars.position)

    if type == 'LINE_COMMENT':
        return 'COMMENT', await line_comment(chars)
    elif type == 'BLOCK_COMMENT':
        return 'COMMENT', await block_comment(chars)
    else:
        return type, value


async def line_comment(chars: Chars) -> str:
    value = '//'
    while chars.next and chars.next not in NEWLINES:
        value += await chars.advance()
    return value


async def block_comment(chars: Chars) -> str:
    value = '/*'
    while not value.endswith('*/'):
        if chars.next:
            value += await chars.advance()
        else:
            raise InvalidCharacter('eof', *chars.position)

    return value
