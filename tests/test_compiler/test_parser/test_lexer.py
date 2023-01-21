from pytest import raises
from typing import Awaitable, Callable, Optional, TypeVar

from src.compiler.parser import lexer
from src.utils import aiter

T = TypeVar('T')
async def _test(
        func: Callable[..., Awaitable[T]], input: list[str], *args, output: Optional[T] = None) -> bool:
    return (await func(await lexer.Chars.new(aiter(input)), *args)) == output


class TestLexNewline:
    async def test_accepts_lf(self):
        assert await _test(lexer.lex_newline, ['\n'], output=('NEWLINE', '\n'))

    async def test_accepts_cr(self):
        assert await _test(lexer.lex_newline, ['\r'], output=('NEWLINE', '\r'))

    async def test_accepts_crlf(self):
        assert await _test(lexer.lex_newline, ['\r\n'], output=('NEWLINE', '\r\n'))


class TestLexWhitespace:
    pass


class TestLexNumber:
    async def test_0b_produces_binary(self):
        assert await _test(lexer.lex_number, ['0b0'], output=('BINARY', '0b0'))

    async def test_binary_only_accepts_01(self):
        assert await _test(lexer.lex_number, ['0b012'], output=('BINARY', '0b01'))

    async def test_0o_produces_octal(self):
        assert await _test(lexer.lex_number, ['0o0'], output=('OCTAL', '0o0'))

    async def test_octal_only_accepts_octdigits(self):
        assert await _test(lexer.lex_number, ['0o012345678'], output=('OCTAL', '0o01234567'))

    async def test_0x_produces_hexadecimal(self):
        assert await _test(lexer.lex_number, ['0x0'], output=('HEXADECIMAL', '0x0'))

    async def test_hexadecimal_only_accepts_hexdigits(self):
        assert await _test(lexer.lex_number, ['0x0123456789abcdefg'], output=('HEXADECIMAL', '0x0123456789abcdef'))

    async def test_integer_only_accepts_0_to_9(self):
        assert await _test(lexer.lex_number, ['0123456789a'], output=('INTEGER', '0123456789'))

    async def test_decimal_point_produces_real(self):
        assert await _test(lexer.lex_number, ['0.0'], output=('NUMBER', '0.0'))

    async def test_decimal_point_must_be_followed_by_digit(self):
        with raises(lexer.InvalidCharacter):
            await _test(lexer.lex_number, ['0._'])

    async def test_fractional_part_accepts_0_to_9(self):
        assert await _test(lexer.lex_number, ['0.0123456789a'], output=('NUMBER', '0.0123456789'))

    async def test_exponent_produces_real(self):
        assert await _test(lexer.lex_number, ['0e0'], output=('NUMBER', '0e0'))

    async def test_exponent_must_contain_digit(self):
        with raises(lexer.InvalidCharacter):
            await _test(lexer.lex_number, ['0e_'])

    async def test_exponent_accepts_0_to_9(self):
        assert await _test(lexer.lex_number, ['0e0123456789a'], output=('NUMBER', '0e0123456789'))

    async def test_exponent_may_contain_sign(self):
        assert await _test(lexer.lex_number, ['0e+0'], output=('NUMBER', '0e+0'))
        assert await _test(lexer.lex_number, ['0e-0'], output=('NUMBER', '0e-0'))

    async def test_decimal_may_end_with_imaginary_unit(self):
        assert await _test(lexer.lex_number, ['0j'], output=('IMAGINARY_INTEGER', '0j'))
        assert await _test(lexer.lex_number, ['0.0j'], output=('IMAGINARY_NUMBER', '0.0j'))
        assert await _test(lexer.lex_number, ['0e0j'], output=('IMAGINARY_NUMBER', '0e0j'))


class TestDigits:
    async def test_underscore_must_be_followed_by_alphabet_character(self):
        with raises(lexer.InvalidCharacter):
            await _test(lexer.digits, ['_8'], set('01234567'))

    async def test_accepts_underscore_and_alphabet_characters(self):
        assert await _test(lexer.digits, ['_012345678'], set('01234567'), output='_01234567')


class TestLexName:
    async def test_accepts_ascii_letters_digits_and_underscore(self):
        assert await _test(lexer.lex_name, ['_abc012.'], output=('NAME', '_abc012'))

    async def test_keyword_has_kw_prefixed_kind(self):
        assert await _test(lexer.lex_name, ['assert'], output=('KW_ASSERT', 'assert'))


class TestLexString:
    async def test_string_begins_and_ends_with_same_quote(self):
        assert await _test(lexer.lex_string, ['"foo"'], output=('STRING', '"foo"'))
        assert await _test(lexer.lex_string, ["'foo'"], output=('STRING', "'foo'"))
        with raises(lexer.InvalidCharacter):
            await _test(lexer.lex_string, ['\'foo"'])

    async def test_string_cannot_contain_newline(self):
        with raises(lexer.InvalidCharacter):
            await _test(lexer.lex_string, ['"foo\n', 'bar"'])

    async def test_any_character_can_be_escaped(self):
        assert await _test(lexer.lex_string, ['"foo\\\n', 'bar"'], output=('STRING', '"foo\\\nbar"'))


class TestLexPunctuation:
    async def test_operator_has_op_prefixed_kind(self):
        assert await _test(lexer.lex_punctuation, ['?'], output=('OP_MAYBE', '?'))

    async def test_delimiter_has_no_prefix(self):
        assert await _test(lexer.lex_punctuation, ['{'], output=('LBRACE', '{'))

    async def test_longest_possible_match_is_made(self):
        assert await _test(lexer.lex_punctuation, ['>>='], output=('OP_RSHIFTEQ', '>>='))

    async def test_line_comment_triggers_comment_match(self):
        assert await _test(lexer.lex_punctuation, ['// foo bar'], output=('COMMENT', '// foo bar'))

    async def test_block_comment_triggers_comment_match(self):
        assert await _test(lexer.lex_punctuation, ['/* foo bar */'], output=('COMMENT', '/* foo bar */'))


class TestLineComment:
    async def test_consumes_until_newline_or_eof(self):
        assert await _test(lexer.line_comment, [' foo bar\n'], output='// foo bar')
        assert await _test(lexer.line_comment, [' foo bar'], output='// foo bar')


class TestBlockComment:
    async def test_consumes_until_comment_close(self):
        assert await _test(lexer.block_comment, [' foo\n', 'bar\n', 'baz */\n'], output='/* foo\nbar\nbaz */')

    async def test_must_have_comment_close(self):
        with raises(lexer.InvalidCharacter):
            await _test(lexer.block_comment, [' foo'])
