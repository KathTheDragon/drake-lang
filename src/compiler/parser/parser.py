from collections import deque
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass, field, InitVar
from typing import AsyncIterator, Callable, ClassVar, TypeVar

from . import lexer

# Exceptions
class UnexpectedToken(Exception):
    def __init__(self, token: lexer.Token, expected: Sequence[str] = (), message: str = '') -> None:
        if expected and not message:
            message = f'expected {", ".join(map(repr, expected))}, got'
        super().__init__(f'{message} {token}'.strip())


# Helpers
EOF = lexer.Token('EOF', 'eof')
@dataclass
class Tokens:
    lines: InitVar[AsyncIterator[str]]
    _iterator: AsyncIterator[lexer.Token] = field(init=False)
    _next: deque[lexer.Token] = field(init=False, default_factory=lambda: deque())
    last: lexer.Token = lexer.NULL

    def __post_init__(self, lines: AsyncIterator[str]) -> None:
        self._iterator = lexer.lex(lines)

    async def peek(self, *kinds: str, index: int = 0) -> lexer.Token:
        while len(self._next) <= index:
            self._next.append(await self._iterator.__anext__())
        if not kinds or self._next[index].kind in kinds:
            return self._next[index]
        else:
            return lexer.NULL

    async def maybe(self, *kinds: str) -> lexer.Token:
        try:
            return await self.pop(*kinds)
        except UnexpectedToken:
            return lexer.NULL

    async def pop(self, *kinds: str, message: str = '') -> lexer.Token:
        if next := await self.peek(*kinds):
            self.last = next
            return self._next.popleft()
        else:
            raise UnexpectedToken(await self.peek(), kinds, message)

    def push(self, token: lexer.Token) -> None:
        self._next.appendleft(token)


class Node:
    startpos: lexer.Position
    endpos: lexer.Position


T = TypeVar('T', bound=Node)
ParserCoro = Callable[..., Coroutine[None, None, T]]
async def _itemlist(
        tokens: Tokens, itemcoro: ParserCoro[T], *lookahead: str, initial: bool = True, optional: bool = True,
        separator: str = '', **kw) -> list[T]:
    items = []
    if optional and initial and await tokens.peek(*lookahead):
        return items
    else:
        if initial:
            items.append(await itemcoro(tokens, **kw))
        if not separator:
            separator = (await tokens.peek('COMMA', 'NEWLINE')).kind
        while True:
            if await tokens.maybe(separator):
                if separator == 'COMMA':
                    await tokens.maybe('NEWLINE')
                if lookahead and await tokens.peek(*lookahead):
                    return items
                else:
                    items.append(await itemcoro(tokens, **kw))
            elif await tokens.peek(*lookahead):
                return items
            else:
                if separator:
                    raise UnexpectedToken(await tokens.peek(), (*lookahead, separator))
                else:
                    raise UnexpectedToken(await tokens.peek(), (*lookahead, 'COMMA', 'NEWLINE'))


async def _bracketed_list(tokens: Tokens, itemcoro: ParserCoro[T], left: str, right: str) -> list[T]:
    await tokens.pop(left)
    items = await _itemlist(tokens, itemcoro, right)
    await tokens.pop(right)
    return items


async def _switch(tokens: Tokens, table: dict[str, ParserCoro[T]], default: ParserCoro[T] | None = None, **kw) -> T:
    coro = table.get((await tokens.peek()).kind, default)
    if coro is not None:
        return await coro(tokens, **kw)
    else:
        raise UnexpectedToken(await tokens.peek(), table)


async def _optional(
        tokens: Tokens, kind: str, itemcoro: ParserCoro[T], advance: bool = True) -> T | None:
    if await tokens.peek(kind):
        if advance:
            await tokens.pop()
        return await itemcoro(tokens)
    else:
        return None


# Nodes
KEYWORDS = [f'KW_{kw.upper()}' for kw in lexer.KEYWORDS]
@dataclass
class Expression(Node):
    @staticmethod
    async def parse(tokens: Tokens) -> 'Expression':
        if await tokens.peek(*KEYWORDS):
            return await Keyword.parse(tokens)
        else:
            return await InfixOp.parse(tokens)


@dataclass
class Primary(Expression):
    @staticmethod
    async def parse(tokens: Tokens) -> 'Primary':
        obj = await Atom.parse(tokens)
        primary_table: dict[str, ParserCoro[Primary]] = {
            'DOT': Attribute.parse,
            'LBRACKET': Call.parse,
            'LSQUARE': Index.parse,
        }
        while coro := primary_table.get((await tokens.peek()).kind):
            obj = await coro(tokens, obj)
        return obj


@dataclass
class Atom(Primary):
    @staticmethod
    async def parse(tokens: Tokens) -> 'Atom':
        atom_table: dict[str, ParserCoro[Atom]] = {
            'LBRACE': parse_block_or_mapping,
            'LSQUARE': List.parse,
            'LBRACKET': Grouping.parse,
            'NAME': Name.parse,
        }
        return await _switch(tokens, atom_table, default=Literal.parse)


@dataclass
class Name(Atom):
    token: lexer.Token

    @staticmethod
    async def parse(tokens: Tokens) -> 'Name':
        return Name(await tokens.pop('NAME'))


NUMBERS = [
    'BINARY',
    'OCTAL',
    'HEXADECIMAL',
    'INTEGER',
    'NUMBER',
    'IMAGINARY_INTEGER',
    'IMAGINARY_NUMBER',
]
BOOLEANS = [
    'KW_TRUE',
    'KW_FALSE'
]
LITERALS = [
    'STRING',
    *NUMBERS,
    *BOOLEANS,
    'KW_NONE',
    'KW_BREAK',
    'KW_CONTINUE'
]
@dataclass
class Literal(Atom):
    token: lexer.Token

    @staticmethod
    async def parse(tokens: Tokens, *kinds: str) -> 'Literal':
        if kinds:
            if kind := next((kind for kind in kinds if kind not in LITERALS), ''):
                raise ValueError(f'{kind!r} is not a literal kind')
        else:
            kinds = LITERALS
        return Literal(await tokens.pop(*kinds))


@dataclass
class Block(Atom):
    expressions: list[Expression]

    @staticmethod
    async def parse(tokens: Tokens, initial: Expression | None = None) -> 'Block':
        if initial is None:
            await tokens.pop('LBRACE')
            items = []
        else:
            items = [initial]
        items.extend(await _itemlist(tokens, Expression.parse, 'RBRACE', initial=not items))
        await tokens.pop('RBRACE')
        return Block(items)


@dataclass
class StarExpression(Node):
    expression: Expression

    @staticmethod
    async def parse(tokens: Tokens) -> 'StarExpression':
        await tokens.pop('OP_MULT')
        return StarExpression(await Expression.parse(tokens))


@dataclass
class Pair(Node):
    key: Expression
    value: Expression

    @staticmethod
    async def parse(tokens: Tokens) -> 'Pair':
        key = await Expression.parse(tokens)
        await tokens.pop('COLON')
        return Pair(key, await Expression.parse(tokens))


async def parse_mapping_item(tokens: Tokens) -> StarExpression | Pair:
    if await tokens.peek('OP_MULT'):
        return await StarExpression.parse(tokens)
    else:
        return await Pair.parse(tokens)


async def parse_expression_or_pair(tokens: Tokens) -> Expression | Pair:
    expression = await Expression.parse(tokens)
    if await tokens.maybe('COLON'):
        return Pair(expression, await Expression.parse(tokens))
    else:
        return expression


@dataclass
class Mapping(Atom):
    pairs: list[StarExpression | Pair]

    @staticmethod
    async def parse(tokens: Tokens, initial: Pair | StarExpression | None = None) -> 'Mapping':
        if initial is None:
            await tokens.pop('LBRACE')
            items = []
        else:
            items = [initial]
        items.extend(await _itemlist(tokens, Pair.parse, 'RBRACE', initial=not items))
        await tokens.pop('RBRACE')
        return Mapping(items)


async def parse_block_or_mapping(tokens: Tokens) -> Block | Mapping:
    if await tokens.peek('RBRACE', 'OP_MULT', index=1):
        return await Mapping.parse(tokens)
    else:
        await tokens.pop('LBRACE')
        expression = await parse_expression_or_pair(tokens)
        if isinstance(expression, Pair):
            return await Mapping.parse(tokens, expression)
        else:
            return await Block.parse(tokens, expression)


@dataclass
class Range(Node):
    start: Expression
    inclusive: bool
    end: Expression
    step: Expression | None

    @staticmethod
    async def parse(tokens: Tokens, start: Expression | None = None) -> 'Range':
        if start is None:
            start = await Expression.parse(tokens)
        await tokens.pop('OP_RANGE')
        if await tokens.maybe('OP_EQ'):
            inclusive = True
            end = await Expression.parse(tokens)
        elif not await tokens.peek('COLON', 'COMMA', 'NEWLINE', 'RSQUARE'):
            inclusive = False
            end = await Expression.parse(tokens)
        else:
            inclusive = False
            end = None
        step = await _optional(tokens, 'COLON', PrefixOp.parse)
        return Range(start, inclusive, end, step)


async def parse_listitem(tokens: Tokens) -> StarExpression | Range | Expression:
    if await tokens.peek('OP_MULT'):
        return await StarExpression.parse(tokens)
    else:
        expression = await Expression.parse(tokens)
        if await tokens.peek('OP_RANGE'):
            return await Range.parse(tokens, start=expression)
        else:
            return expression


@dataclass
class List(Atom):
    items: list[StarExpression | Range | Expression]

    @staticmethod
    async def parse(tokens: Tokens) -> 'List':
        return List(await _bracketed_list(tokens, parse_listitem, 'LSQUARE', 'RSQUARE'))


@dataclass
class LabelledExpression(Node):
    label: Name | None
    expression: Expression

    @staticmethod
    async def parse(tokens: Tokens) -> 'LabelledExpression':
        # extend _optional for this?
        if await tokens.peek('NAME') and await tokens.peek('COLON', index=1):
            name = await Name.parse(tokens)
            await tokens.maybe('COLON')
        else:
            name = None
        return LabelledExpression(name, await Expression.parse(tokens))


@dataclass
class Grouping(Atom):
    items: list[LabelledExpression]

    @staticmethod
    async def parse(tokens: Tokens) -> 'Grouping':
        return Grouping(await _bracketed_list(tokens, LabelledExpression.parse, 'LBRACKET', 'RBRACKET'))


@dataclass
class Attribute(Primary):
    object: Primary
    attribute: Name

    @staticmethod
    async def parse(tokens: Tokens, obj: Primary | None = None) -> 'Attribute':
        if obj is None:
            obj = await Atom.parse(tokens)
        await tokens.pop('DOT')
        return Attribute(obj, await Name.parse(tokens))


async def parse_arg(tokens: Tokens) -> StarExpression | LabelledExpression:
    if await tokens.peek('OP_MULT'):
        return await StarExpression.parse(tokens)
    else:
        return await LabelledExpression.parse(tokens)


@dataclass
class Call(Primary):
    object: Primary
    args: list[LabelledExpression | StarExpression]

    @staticmethod
    async def parse(tokens: Tokens, obj: Primary | None = None) -> 'Call':
        if obj is None:
            obj = await Atom.parse(tokens)
        return Call(obj, await _bracketed_list(tokens, parse_arg, 'LBRACKET', 'RBRACKET'))


@dataclass
class Index(Primary):
    object: Primary
    index: List

    @staticmethod
    async def parse(tokens: Tokens, obj: Primary | None = None) -> 'Index':
        if obj is None:
            obj = await Atom.parse(tokens)
        return Index(obj, await List.parse(tokens))


POSTFIX_OPERATORS = [
    'OP_MAYBE',
]
@dataclass
class PostfixOp(Expression):
    expression: 'PostfixOp | Primary'
    operator: lexer.Token

    @staticmethod
    async def parse(tokens: Tokens, expression: 'PostfixOp | Primary | None' = None) -> 'PostfixOp | Primary':
        if expression is None:
            expression = await Primary.parse(tokens)
        while operator := (await tokens.maybe(*POSTFIX_OPERATORS)):
            expression = PostfixOp(expression, operator)
        return expression


PREFIX_OPERATORS = [
    'KW_NOT',
    'OP_INV',
    'OP_ADD',
    'OP_SUB',
]
@dataclass
class PrefixOp(Expression):
    operator: lexer.Token
    expression: 'PrefixOp | PostfixOp | Primary'

    @staticmethod
    async def parse(tokens: Tokens) -> 'PrefixOp | PostfixOp | Primary':
        if operator := (await tokens.maybe(*PREFIX_OPERATORS)):
            return PrefixOp(operator, await PrefixOp.parse(tokens))
        else:
            return await PostfixOp.parse(tokens)


PRECEDENCE = {
    '**': 17,
    '%':  16,
    '*':  16,
    '/':  16,
    '+':  15,
    '-':  15,
    '<<': 14,
    '>>': 14,
    '!&': 13,
    '&':  12,
    '!^': 11,
    '^':  10,
    '!|': 9,
    '|':  8,
    '<':  7,
    '!<': 7,
    '<=': 7,
    '!<=': 7,
    '>':  7,
    '!>': 7,
    '>=': 7,
    '!>=': 7,
    '==': 7,
    '!=': 7,
    'is': 7,
    'is not': 7,
    'is in': 7,
    'is not in': 7,
    'nand': 6,
    'and': 5,
    'xnor': 4,
    'xor': 3,
    'nor': 2,
    'or': 1,
    '': 0  # For reasons
}
RIGHT = [
    '**',
    '->'
]
BINARY_OPERATORS = [
    'OP_ADD',
    'OP_SUB',
    'OP_MULT',
    'OP_DIV',
    'OP_MOD',
    'OP_EQ',
    'OP_NEQ',
    'OP_LT',
    'OP_LE',
    'OP_GT',
    'OP_GE',
    'OP_IS',
    'OP_ISNOT',
    'OP_IN',
    'OP_NOTIN',
    'OP_OR',
    'OP_NOR',
    'OP_XOR',
    'OP_XNOR',
    'OP_AND',
    'OP_NAND',
    'OP_LSHIFT',
    'OP_RSHIFT',
    'OP_ARROW',
]
async def _branch_operator(tokens: Tokens, operator: str = '') -> lexer.Token:
    if token_is := await tokens.maybe('KW_IS'):
        if token_not := await tokens.maybe('KW_NOT'):
            if token_in := await tokens.maybe('KW_IN'):
                tokens.push(lexer.Token('OP_NOTIN', 'is not in', token_is.start, token_in.end))
            else:
                tokens.push(lexer.Token('OP_ISNOT', 'is not', token_is.start, token_not.end))
        elif token_in := await tokens.maybe('KW_IN'):
            tokens.push(lexer.Token('OP_IN', 'is in', token_is.start, token_in.end))
        else:
            tokens.push(lexer.Token('OP_IS', 'is', token_is.start, token_is.end))

    next_op = (await tokens.peek(*BINARY_OPERATORS)).value

    if PRECEDENCE[operator] < PRECEDENCE[next_op] or PRECEDENCE[operator] == PRECEDENCE[next_op] and next_op in RIGHT:
        return await tokens.pop()
    else:
        return lexer.NULL


@dataclass
class InfixOp(Expression):
    left: 'InfixOp | PrefixOp | PostfixOp | Primary'
    operator: lexer.Token
    right: 'InfixOp | PrefixOp | PostfixOp | Primary'

    @staticmethod
    async def parse(
            tokens: Tokens, left: PrefixOp | PostfixOp | Primary | None = None, operator: lexer.Token = lexer.NULL
            ) -> 'InfixOp | PrefixOp | PostfixOp | Primary':
        if left is None:
            left = await PrefixOp.parse(tokens)
        while next_op := await _branch_operator(tokens, operator.value):
            left = InfixOp(left, next_op, await InfixOp.parse(tokens, operator=next_op))
        return left


@dataclass
class PatternBranch(Node):
    @staticmethod
    async def parse(tokens: Tokens) -> 'PatternBranch':
        pattern_table: dict[str, ParserCoro[PatternBranch]] = {
            'OP_EQ': PatternValue.parse,
            'LBRACE': PatternMapping.parse,
            'LSQUARE': PatternList.parse,
            'LBRACKET': PatternGrouping.parse,
            'NAME': parse_pattern_instance_or_binding,
        }
        return await _switch(tokens, pattern_table, default=PatternLiteral.parse)


@dataclass
class Pattern(Node):
    branches: list[PatternBranch]
    binding: Name | None

    @staticmethod
    async def parse(tokens: Tokens) -> 'Pattern':
        return Pattern(
            await _itemlist(tokens, PatternBranch.parse, optional=False, separator='OP_OR'),
            await _optional(tokens, 'KW_AS', Name.parse))


@dataclass
class LabelledPattern(Node):
    name: Name | None
    pattern: Pattern

    @staticmethod
    async def parse(tokens: Tokens) -> 'LabelledPattern':
        # extend _optional for this?
        if await tokens.peek('NAME') and await tokens.peek('COLON', index=1):
            name = await Name.parse(tokens)
            await tokens.maybe('COLON')
        else:
            name = None
        return LabelledPattern(name, await Pattern.parse(tokens))


@dataclass
class PatternValue(PatternBranch):
    value: Primary

    @staticmethod
    async def parse(tokens: Tokens) -> 'PatternValue':
        await tokens.pop('OP_EQ')
        return PatternValue(await Primary.parse(tokens))


@dataclass
class PatternInstance(PatternBranch):
    name: Name
    params: list[LabelledPattern]

    @staticmethod
    async def parse(tokens: Tokens) -> 'PatternInstance':
        return PatternInstance(
            await Name.parse(tokens), await _bracketed_list(tokens, LabelledPattern.parse, 'LBRACKET', 'RBRACKET'))


@dataclass
class PatternBinding(PatternBranch):
    binding: Name | Attribute

    @staticmethod
    async def parse(tokens: Tokens) -> 'PatternBinding':
        binding = await Name.parse(tokens)
        while await tokens.maybe('DOT'):
            binding = Attribute(binding, await Name.parse(tokens))
        return PatternBinding(binding)


async def parse_pattern_instance_or_binding(tokens: Tokens) -> PatternInstance | PatternBinding:
    if await tokens.peek('NAME') and await tokens.peek('LBRACKET', index=1):
        return await PatternInstance.parse(tokens)
    else:
        return await PatternBinding.parse(tokens)


@dataclass
class PatternStarBinding(Node):
    binding: PatternBinding

    @staticmethod
    async def parse(tokens: Tokens) -> 'PatternStarBinding':
        await tokens.pop('OP_MULT')
        return PatternStarBinding(await PatternBinding.parse(tokens))


@dataclass
class PatternPair(Node):
    key: Pattern
    value: Pattern

    @staticmethod
    async def parse(tokens: Tokens) -> 'PatternPair':
        key = await Pattern.parse(tokens)
        await tokens.pop('COLON')
        return PatternPair(key, await Pattern.parse(tokens))


async def parse_pattern_mapping_item(tokens: Tokens) -> PatternPair | PatternStarBinding:
    if await tokens.peek('OP_MULT'):
        return await PatternStarBinding.parse(tokens)
    else:
        return await PatternPair.parse(tokens)


@dataclass
class PatternMapping(PatternBranch):
    items: list[PatternPair | PatternStarBinding]

    @staticmethod
    async def parse(tokens: Tokens) -> 'PatternMapping':
        return PatternMapping(await _bracketed_list(tokens, parse_pattern_mapping_item, 'LBRACE', 'RBRACE'))


async def parse_pattern_listitem(tokens: Tokens) -> Pattern | PatternStarBinding:
    if await tokens.peek('OP_MULT'):
        return await PatternStarBinding.parse(tokens)
    else:
        return await Pattern.parse(tokens)


@dataclass
class PatternList(PatternBranch):
    items: list[Pattern | PatternStarBinding]

    @staticmethod
    async def parse(tokens: Tokens) -> 'PatternList':
        return PatternList(await _bracketed_list(tokens, parse_pattern_listitem, 'LSQUARE', 'RSQUARE'))


@dataclass
class PatternGrouping(PatternBranch):
    items: list[LabelledPattern]

    @staticmethod
    async def parse(tokens: Tokens) -> 'PatternGrouping':
        return PatternGrouping(await _bracketed_list(tokens, LabelledPattern.parse, 'LBRACKET', 'RBRACKET'))


async def parse_number(tokens: Tokens) -> Literal | PrefixOp | InfixOp:
    operator = await tokens.maybe('OP_ADD', 'OP_SUB')
    number = await Literal.parse(tokens, *NUMBERS)
    if operator:
        number = PrefixOp(operator, number)
    if tokens.last.kind in ('INTEGER', 'NUMBER'):
        if operator := await tokens.maybe('OP_ADD', 'OP_SUB'):
            imaginary = await Literal.parse(tokens, 'IMAGINARY_INTEGER', 'IMAGINARY_NUMBER')
            number = InfixOp(number, operator, imaginary)
    return number


@dataclass
class PatternLiteral(PatternBranch):
    literal: Literal | PrefixOp | InfixOp

    @staticmethod
    async def parse(tokens: Tokens) -> 'PatternLiteral':
        if await tokens.peek('OP_ADD', 'OP_SUB', *NUMBERS):
            literal = await parse_number(tokens)
        else:
            literal = await Literal.parse(tokens)
        return PatternLiteral(literal)


@dataclass
class Keyword(Expression):
    @staticmethod
    async def parse(tokens: Tokens) -> 'Keyword':
        kw_table: dict[str, ParserCoro[Keyword]] = {
            'KW_LET': Declaration.parse,
            'KW_VAR': Declaration.parse,
            'KW_IF': If.parse,
            'KW_MATCH': Match.parse,
            'KW_TRY': Try.parse,
            'KW_THROW': Throw.parse,
            'KW_FOR': For.parse,
            'KW_WHILE': While.parse,
            'KW_FUNC': Func.parse,
            'KW_TYPE': Type.parse,
            'KW_INTERFACE': Interface.parse,
            'KW_EXCEPTION': ExceptionType.parse,
            'KW_ENUM': Enum.parse,
            'KW_MODULE': Module.parse,
            'KW_ITER': Iter.parse,
            'KW_MUTABLE': Mutable.parse,
            'KW_ASSERT': Assert.parse,
            'KW_DEFER': Defer.parse,
            'KW_IMPORT': Import.parse,
        }
        return await _switch(tokens, kw_table)


@dataclass
class Typehint(Node):
    expression: Expression

    @staticmethod
    async def parse(tokens: Tokens) -> 'Typehint':
        await tokens.pop('OP_LT')
        expression = await Expression.parse(tokens)
        await tokens.pop('OP_GT')
        return Typehint(expression)


ASSIGNMENT_OPERATORS = [
    'OP_EQ',
    'OP_ADDEQ',
    'OP_SUBEQ',
    'OP_MULTEQ',
    'OP_DIVEQ',
    'OP_MODEQ',
    'OP_POWEQ',
    'OP_OREQ',
    'OP_NOREQ',
    'OP_XOREQ',
    'OP_XNOREQ',
    'OP_ANDEQ',
    'OP_NANDEQ',
    'OP_LSHIFTEQ',
    'OP_RSHIFTEQ',
]
@dataclass
class Declaration(Keyword):
    variable: bool
    typehint: Typehint | None
    pattern: Pattern
    operator: lexer.Token | None
    expression: Expression | None

    @staticmethod
    async def parse(tokens: Tokens) -> 'Declaration':
        variable = (await tokens.pop('KW_LET', 'KW_VAR')).kind == 'KW_VAR'
        typehint = await _optional(tokens, 'OP_LT', Typehint.parse, advance=False)
        pattern = await Pattern.parse(tokens)
        operator = await tokens.peek(*ASSIGNMENT_OPERATORS)
        return Declaration(
            variable, typehint, pattern, operator, await _optional(tokens, operator.kind, Expression.parse))


@dataclass
class If(Keyword):
    condition: Expression
    result: Expression
    default: Expression | None

    @staticmethod
    async def parse(tokens: Tokens) -> 'If':
        await tokens.pop('KW_IF')
        condition = await Expression.parse(tokens)
        await tokens.pop('KW_THEN')
        return If(condition, await Expression.parse(tokens), await _optional(tokens, 'KW_ELSE', Expression.parse))


@dataclass
class Case(Node):
    pattern: Pattern
    condition: Expression | None
    expression: Expression

    @staticmethod
    async def parse(tokens: Tokens) -> 'Case':
        pattern = await Pattern.parse(tokens)
        condition = await _optional(tokens, 'KW_IF', Expression.parse)
        await tokens.pop('COLON')
        return Case(pattern, condition, await Expression.parse(tokens))


@dataclass
class Match(Keyword):
    expression: Expression
    cases: list[Case]
    default: Expression | None

    @staticmethod
    async def parse(tokens: Tokens) -> 'Match':
        await tokens.pop('KW_MATCH')
        expression = await Expression.parse(tokens)
        await tokens.pop('KW_IN')
        return Match(
            expression,
            await _bracketed_list(tokens, Case.parse, 'LBRACE', 'RBRACE'),
            await _optional(tokens, 'KW_ELSE', Expression.parse))


@dataclass
class Try(Keyword):
    expression: Expression
    cases: list[Case]
    default: Expression | None

    @staticmethod
    async def parse(tokens: Tokens) -> 'Try':
        await tokens.pop('KW_TRY')
        expression = await Expression.parse(tokens)
        await tokens.pop('KW_CATCH')
        return Try(
            expression,
            await _bracketed_list(tokens, Case.parse, 'LBRACE', 'RBRACE'),
            await _optional(tokens, 'KW_ELSE', Expression.parse))


@dataclass
class LoopVar(Node):
    pattern: Pattern
    container: Expression
    condition: Expression | None

    @staticmethod
    async def parse(tokens: Tokens) -> 'LoopVar':
        pattern = await Pattern.parse(tokens)
        await tokens.pop('KW_IN')
        return LoopVar(pattern, await Expression.parse(tokens), await _optional(tokens, 'KW_IF', Expression.parse))


@dataclass
class For(Keyword):
    loopvars: list[LoopVar]
    yieldfrom: bool
    expression: Expression

    @staticmethod
    async def parse(tokens: Tokens) -> 'For':
        await tokens.pop('KW_FOR')
        loopvars = await _itemlist(tokens, LoopVar.parse, 'KW_YIELD')
        await tokens.pop('KW_YIELD')
        return For(loopvars, bool(await tokens.maybe('KW_FROM')), await Expression.parse(tokens))


@dataclass
class While(Keyword):
    condition: Expression
    yieldfrom: bool
    expression: Expression

    @staticmethod
    async def parse(tokens: Tokens) -> 'While':
        await tokens.pop('KW_WHILE')
        condition = await Expression.parse(tokens)
        await tokens.pop('KW_YIELD')
        return While(condition, bool(await tokens.maybe('KW_FROM')), await Expression.parse(tokens))


@dataclass
class Throw(Keyword):
    exception: Expression

    @staticmethod
    async def parse(tokens: Tokens) -> 'Throw':
        await tokens.pop('KW_THROW')
        return Throw(await Expression.parse(tokens))


@dataclass
class TypeParam(Node):
    variance: lexer.Token
    name: Name
    operator: lexer.Token
    bound: Expression | None

    @staticmethod
    async def parse(tokens: Tokens) -> 'TypeParam':
        variance = await tokens.maybe('OP_ADD', ' OP_SUB', 'OP_EQ')
        name = await Name.parse(tokens)
        operator = await tokens.peek('OP_LT', 'OP_LE')
        return TypeParam(variance, name, operator, await _optional(tokens, operator.kind, Expression.parse))


async def parse_typeparams(tokens: Tokens) -> list[TypeParam]:
    if await tokens.peek('LSQUARE'):
        return await _bracketed_list(tokens, TypeParam.parse, 'LSQUARE', 'RSQUARE')
    else:
        return []


@dataclass
class Param(Node):
    typehint: Typehint | None
    name: Name
    value: Expression | None

    @staticmethod
    async def parse(tokens: Tokens) -> 'Param':
        return Param(
            await Typehint.parse(tokens), await Name.parse(tokens), await _optional(tokens, 'COLON', Expression.parse))


@dataclass
class Func(Keyword):
    typeparams: list[TypeParam]
    params: list[Param]
    body: Expression

    @staticmethod
    async def parse(tokens: Tokens) -> 'Func':
        await tokens.pop('KW_FUNC')
        typeparams = await parse_typeparams(tokens)
        # extend _bracketed_list for this
        await tokens.pop('LBRACKET')
        params = []
        if token := await tokens.maybe('KW_SELF'):
            params.append(Param(None, Name(token), None))
        params.extend(await _itemlist(tokens, Param.parse, 'RBRACKET', initial=not params))
        await tokens.pop('RBRACKET')
        await tokens.pop('OP_ARROW')
        return Func(typeparams, params, await Expression.parse(tokens))


Self = TypeVar('Self', bound='_Type')
@dataclass
class _Type(Keyword):
    keyword: ClassVar[str]
    params: list[TypeParam]
    supertype: Expression | None
    body: Block

    @classmethod
    async def parse(cls: type[Self], tokens: Tokens) -> Self:
        await tokens.pop(cls.keyword)
        return cls(
            await parse_typeparams(tokens),
            await _optional(tokens, 'OP_LT', Expression.parse),
            await Block.parse(tokens))


@dataclass
class Type(_Type):
    keyword = 'KW_TYPE'


@dataclass
class Interface(_Type):
    keyword = 'KW_INTERFACE'


@dataclass
class ExceptionType(_Type):
    keyword = 'KW_EXCEPTION'


@dataclass
class EnumItem(Node):
    name: Name
    value: Expression | None

    @staticmethod
    async def parse(tokens: Tokens) -> 'EnumItem':
        return EnumItem(await Name.parse(tokens), await _optional(tokens, 'COLON', Expression.parse))


@dataclass
class Enum(Keyword):
    @staticmethod
    async def parse(tokens: Tokens) -> 'Enum':
        await tokens.pop('KW_ENUM')
        if await tokens.peek('KW_FLAGS'):
            return await FlagEnum.parse(tokens)
        else:
            return await SimpleEnum.parse(tokens)


@dataclass
class SimpleEnum(Enum):
    params: list[TypeParam]
    supertype: Expression | None
    items: list[EnumItem]

    @staticmethod
    async def parse(tokens: Tokens) -> 'SimpleEnum':
        return SimpleEnum(
            await parse_typeparams(tokens),
            await _optional(tokens, 'OP_LT', Expression.parse),
            await _bracketed_list(tokens, EnumItem.parse, 'LBRACE', 'RBRACE'))


@dataclass
class FlagEnum(Enum):
    items: list[EnumItem]

    @staticmethod
    async def parse(tokens: Tokens) -> 'FlagEnum':
        await tokens.pop('KW_FLAGS')
        return FlagEnum(await _bracketed_list(tokens, EnumItem.parse, 'LBRACE', 'RBRACE'))


@dataclass
class Module(Keyword):
    body: Block

    @staticmethod
    async def parse(tokens: Tokens) -> 'Module':
        await tokens.pop('KW_MODULE')
        return Module(await Block.parse(tokens))


@dataclass
class Iter(Keyword):
    expression: Expression

    @staticmethod
    async def parse(tokens: Tokens) -> 'Iter':
        await tokens.pop('KW_ITER')
        iter_table: dict[str, ParserCoro[List | For | While]] = {
            'LSQUARE': List.parse,
            'KW_FOR': For.parse,
            'KW_WHILE': While.parse,
        }
        return Iter(await _switch(tokens, iter_table))


@dataclass
class Mutable(Keyword):
    expression: Expression

    @staticmethod
    async def parse(tokens: Tokens) -> 'Mutable':
        await tokens.pop('KW_MUTABLE')
        mutable_table: dict[str, ParserCoro[Type | Iter | Mapping | List | Literal]] = {
            'KW_TYPE': Type.parse,
            'KW_ITER': Iter.parse,
            'LBRACE': Mapping.parse,
            'LSQUARE': List.parse,
            'STRING': Literal.parse,
        }
        return Mutable(await _switch(tokens, mutable_table))


@dataclass
class Assert(Keyword):
    expression: Expression
    throws: Expression | None

    @staticmethod
    async def parse(tokens: Tokens) -> 'Assert':
        await tokens.pop('KW_ASSERT')
        return Assert(await Expression.parse(tokens), await _optional(tokens, 'KW_THROWS', Expression.parse))


@dataclass
class Defer(Keyword):
    expression: Expression

    @staticmethod
    async def parse(tokens: Tokens) -> 'Defer':
        await tokens.pop('KW_DEFER')
        return Defer(await Expression.parse(tokens))


@dataclass
class Import(Keyword):
    path: lexer.Token

    @staticmethod
    async def parse(tokens: Tokens) -> 'Import':
        await tokens.pop('KW_IMPORT')
        return Import(await tokens.pop('STRING'))


# Parser functions
async def parse(lines: AsyncIterator[str]) -> Module:
    tokens = Tokens(lines)
    expressions = await _itemlist(tokens, Expression.parse, 'EOF')
    await tokens.pop('EOF')
    return Module(Block(expressions))
