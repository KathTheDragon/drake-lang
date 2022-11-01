import string
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from dataclasses import dataclass, field, InitVar
from typing import TypeVar

KEYWORDS = [
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
]
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

@dataclass
class Token:
    kind: str
    value: str
    start: tuple[int, int] = (-1, -1)
    end: tuple[int, int] = (-1, -1)

    def __post_init__(self) -> None:
        if self.end == (-1, -1) and self.start != (-1, -1):
            self.end = self.start


NULL = Token('', '')

# Exceptions
class InvalidCharacter(Exception):
    def __init__(self, char: str, linenum: int, column: int) -> None:
        super().__init__(f'{char!r} @ {linenum}:{column}')


class UnexpectedToken(Exception):
    def __init__(self, token: Token, expected: Sequence[str] = (), message: str = '') -> None:
        if expected and not message:
            message = f'expected {", ".join(map(repr, expected))}, got'
        super().__init__(f'{message} {token}'.strip())


T = TypeVar('T')
async def aenumerate(aiterable: AsyncIterable[T], start: int = 0) -> AsyncIterator[tuple[int, T]]:
    count = start
    async for item in aiterable:
        yield count, item
        count += 1


@dataclass
class Chars:
    lines: InitVar[AsyncIterator[str]]
    _iterator: AsyncIterator[tuple[str, tuple[int, int]]] = field(init=False)
    next: str = field(init=False)
    position: tuple[int, int] = field(init=False)

    def __post_init__(self, lines: AsyncIterator[str]) -> None:
        self._iterator = self._iterate(lines)

    @staticmethod
    async def new(lines: AsyncIterator[str]) -> 'Chars':
        self = Chars(lines)
        self._next, self._position = await self._iterator.__anext__()
        return self

    @staticmethod
    async def _iterate(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str, tuple[int, int]]]:
        linenum, column = 1, 0
        async for linenum, line in aenumerate(lines, start=linenum):
            column = 0
            for column, char in enumerate(line, start=column):
                yield char, (linenum, column)
        while True:
            yield '', (linenum, column + 1)

    async def advance(self) -> str:
        next = self.next
        self.next, self.position = await self._iterator.__anext__()
        return next


async def lex(lines: AsyncIterator[str]) -> AsyncIterator[Token]:
    chars = await Chars.new(lines)
    while True:
        char = chars.next
        if not char:
            yield Token('EOF', 'eof', chars.position)
        elif char in NEWLINES:
            yield await lex_newline(chars)
        elif char in WHITESPACE:
            yield await lex_whitespace(chars)
        elif char in string.digits:
            yield await lex_number(chars)
        elif char in string.ascii_letters or char == '_':
            yield await lex_name(chars)
        elif char in QUOTES:
            yield await lex_string(chars)
        elif char in PUNCTUATION:
            yield await lex_punctuation(chars)
        else:
            raise InvalidCharacter(char, *chars.position)


async def lex_newline(chars: Chars) -> Token:
    start = chars.position
    char = await chars.advance()
    if char == '\n':
        return Token('NEWLINE', '\n', start, chars.position)
    else:
        char = chars.next
        if char == '\n':
            await chars.advance()
            return Token('NEWLINE', '\r\n', start, chars.position)
        else:
            return Token('NEWLINE', '\r', start, chars.position)


async def lex_whitespace(chars: Chars) -> Token:
    value = ''
    start = chars.position
    while chars.next and chars.next in WHITESPACE:
        value += chars.next
        await chars.advance()
    return Token('WHITESPACE', value, start, chars.position)


async def lex_number(chars: Chars) -> Token:
    start = chars.position
    value = await chars.advance()
    if value == '0' and chars.next in INTEGER_TYPES:
        prefix = await chars.advance()
        value += prefix
        alphabet, kind = INTEGER_TYPES[prefix]
        value += await digits(chars, alphabet)
    else:
        value += await digits(chars)
        if chars.next == '.':
            value += await chars.advance()
            value += await digits(chars, lead=True)
        if chars.next in ('e', 'E'):
            value += await chars.advance()
            if chars.next in ('+', '-'):
                value += await chars.advance()
            value += await digits(chars, lead=True)
        if chars.next in ('j', 'J'):
            value += await chars.advance()
            kind = 'IMAGINARY'
        else:
            kind = 'REAL'
    return Token(kind, value, start, chars.position)


async def digits(chars: Chars, alphabet: set = set(string.digits), lead: bool = False) -> str:
    if not lead:
        value = ''
    elif chars.next in alphabet:
        value = await chars.advance()
    else:
        raise InvalidCharacter(chars.next, *chars.position)

    while True:
        if chars.next == '_':
            value += await chars.advance()
            if chars.next not in alphabet:
                raise InvalidCharacter(chars.next, *chars.position)
        if chars.next in alphabet:
            value += chars.advance()
        else:
            break

    return value


async def lex_name(chars: Chars) -> Token:
    start = chars.position
    value = ''
    alphabet = string.ascii_letters + string.digits + '_'
    while chars.next in alphabet:
        value += await chars.advance()

    if value in KEYWORDS:
        type = f'KW_{value.upper()}'
    else:
        type = 'NAME'
    return Token(type, value, start, chars.position)


async def lex_string(chars: Chars) -> Token:
    start = chars.position
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

    return Token('STRING', value, start, chars.position)


async def lex_punctuation(chars: Chars) -> Token:
    start = chars.position
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
        return await line_comment(chars, start)
    elif type == 'BLOCK_COMMENT':
        return await block_comment(chars, start)
    else:
        return Token(type, value, start, chars.position)


async def line_comment(chars: Chars, start: tuple[int, int]) -> Token:
    value = '//'
    while chars.next and chars.next not in NEWLINES:
        value += await chars.advance()
    return Token('COMMENT', value, start, chars.position)


async def block_comment(chars: Chars, start: tuple[int, int]) -> Token:
    value = '/*'
    while not value.endswith('*/'):
        if chars.next:
            value += await chars.advance()
        else:
            raise InvalidCharacter('eof', *chars.position)

    return Token('COMMENT', value, start, chars.position)
