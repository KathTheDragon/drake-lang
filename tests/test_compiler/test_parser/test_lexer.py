from pytest import raises
from typing import Callable, Coroutine

from src.compiler.parser import lexer
from src.utils import aiter

async def _test(
        func: Callable[[lexer.Chars], Coroutine[None, None, tuple[str, str]]], input: list[str],
        output: tuple[str, str] = ('', '')) -> bool:
    return (await func(await lexer.Chars.new(aiter(input)))) == output

class TestLexNewline:
    async def test_accepts_lf(self):
        chars = await lexer.Chars.new(aiter(['\n']))
        assert (await lexer.lex_newline(chars)) == ('NEWLINE', '\n')

    async def test_accepts_cr(self):
        chars = await lexer.Chars.new(aiter(['\r']))
        assert (await lexer.lex_newline(chars)) == ('NEWLINE', '\r')

    async def test_accepts_crlf(self):
        chars = await lexer.Chars.new(aiter(['\r\n']))
        assert (await lexer.lex_newline(chars)) == ('NEWLINE', '\r\n')


class TestLexWhitespace:
    pass


class TestLexNumber:
    async def test_0b_produces_binary(self):
        chars = await lexer.Chars.new(aiter(['0b0']))
        assert (await lexer.lex_number(chars)) == ('BINARY', '0b0')

    async def test_binary_only_accepts_01(self):
        chars = await lexer.Chars.new(aiter(['0b012']))
        assert (await lexer.lex_number(chars)) == ('BINARY', '0b01')

    async def test_0o_produces_octal(self):
        chars = await lexer.Chars.new(aiter(['0o0']))
        assert (await lexer.lex_number(chars)) == ('OCTAL', '0o0')

    async def test_octal_only_accepts_octdigits(self):
        chars = await lexer.Chars.new(aiter(['0o012345678']))
        assert (await lexer.lex_number(chars)) == ('OCTAL', '0o01234567')

    async def test_0x_produces_hexadecimal(self):
        chars = await lexer.Chars.new(aiter(['0x0']))
        assert (await lexer.lex_number(chars)) == ('HEXADECIMAL', '0x0')

    async def test_hexadecimal_only_accepts_hexdigits(self):
        chars = await lexer.Chars.new(aiter(['0x0123456789abcdefg']))
        assert (await lexer.lex_number(chars)) == ('HEXADECIMAL', '0x0123456789abcdef')

    async def test_integer_only_accepts_0_to_9(self):
        chars = await lexer.Chars.new(aiter(['0123456789a']))
        assert (await lexer.lex_number(chars)) == ('INTEGER', '0123456789')

    async def test_decimal_point_produces_real(self):
        chars = await lexer.Chars.new(aiter(['0.0']))
        assert (await lexer.lex_number(chars)) == ('REAL', '0.0')

    async def test_decimal_point_must_be_followed_by_digit(self):
        chars = await lexer.Chars.new(aiter(['0._']))
        with raises(lexer.InvalidCharacter):
            await lexer.lex_number(chars)

    async def test_fractional_part_accepts_0_to_9(self):
        chars = await lexer.Chars.new(aiter(['0.0123456789a']))
        assert (await lexer.lex_number(chars)) == ('REAL', '0.0123456789')

    async def test_exponent_produces_real(self):
        chars = await lexer.Chars.new(aiter(['0e0']))
        assert (await lexer.lex_number(chars)) == ('REAL', '0e0')

    async def test_exponent_must_contain_digit(self):
        chars = await lexer.Chars.new(aiter(['0e_']))
        with raises(lexer.InvalidCharacter):
            await lexer.lex_number(chars)

    async def test_exponent_accepts_0_to_9(self):
        chars = await lexer.Chars.new(aiter(['0e0123456789a']))
        assert (await lexer.lex_number(chars)) == ('REAL', '0e0123456789')

    async def test_exponent_may_contain_sign(self):
        chars = await lexer.Chars.new(aiter(['0e+0']))
        assert (await lexer.lex_number(chars)) == ('REAL', '0e+0')

        chars = await lexer.Chars.new(aiter(['0e-0']))
        assert (await lexer.lex_number(chars)) == ('REAL', '0e-0')

    async def test_decimal_may_end_with_imaginary_unit(self):
        chars = await lexer.Chars.new(aiter(['0j']))
        assert (await lexer.lex_number(chars)) == ('IMAGINARY_INTEGER', '0j')

        chars = await lexer.Chars.new(aiter(['0.0j']))
        assert (await lexer.lex_number(chars)) == ('IMAGINARY_REAL', '0.0j')

        chars = await lexer.Chars.new(aiter(['0e0j']))
        assert (await lexer.lex_number(chars)) == ('IMAGINARY_REAL', '0e0j')


class TestDigits:
    async def test_underscore_must_be_followed_by_alphabet_character(self):
        chars = await lexer.Chars.new(aiter(['_8']))
        with raises(lexer.InvalidCharacter):
            await lexer.digits(chars, set('01234567'))

    async def test_accepts_underscore_and_alphabet_characters(self):
        chars = await lexer.Chars.new(aiter(['_012345678']))
        assert (await lexer.digits(chars, set('01234567'))) == '_01234567'


class TestLexName:
    async def test_accepts_ascii_letters_digits_and_underscore(self):
        chars = await lexer.Chars.new(aiter(['_abc012.']))
        assert (await lexer.lex_name(chars)) == ('NAME', '_abc012')

    async def test_keyword_has_kw_prefixed_kind(self):
        chars = await lexer.Chars.new(aiter(['assert']))
        assert (await lexer.lex_name(chars)) == ('KW_ASSERT', 'assert')


class TestLexString:
    async def test_string_begins_and_ends_with_same_quote(self):
        assert await _test(lexer.lex_string, ['"foo"'], ('STRING', '"foo"'))
        assert await _test(lexer.lex_string, ["'foo'"], ('STRING', "'foo'"))
        with raises(lexer.InvalidCharacter):
            await _test(lexer.lex_string, ['\'foo"'])

    async def test_string_cannot_contain_newline(self):
        with raises(lexer.InvalidCharacter):
            await _test(lexer.lex_string, ['"foo\n', 'bar"'])

    async def test_any_character_can_be_escaped(self):
        assert await _test(lexer.lex_string, ['"foo\\\n', 'bar"'], ('STRING', '"foo\\\nbar"'))


class TestLexPunctuation:
    async def test_operator_has_op_prefixed_kind(self):
        chars = await lexer.Chars.new(aiter(['?']))
        assert (await lexer.lex_punctuation(chars)) == ('OP_MAYBE', '?')

    async def test_delimiter_has_no_prefix(self):
        chars = await lexer.Chars.new(aiter(['{']))
        assert (await lexer.lex_punctuation(chars)) == ('LBRACE', '{')

    async def test_longest_possible_match_is_made(self):
        chars = await lexer.Chars.new(aiter(['>>=']))
        assert (await lexer.lex_punctuation(chars)) == ('OP_RSHIFTEQ', '>>=')

    async def test_line_comment_triggers_comment_match(self):
        chars = await lexer.Chars.new(aiter(['// foo bar']))
        assert (await lexer.lex_punctuation(chars)) == ('COMMENT', '// foo bar')

    async def test_block_comment_triggers_comment_match(self):
        chars = await lexer.Chars.new(aiter(['/* foo bar */']))
        assert (await lexer.lex_punctuation(chars)) == ('COMMENT', '/* foo bar */')


class TestLineComment:
    async def test_consumes_until_newline_or_eof(self):
        chars = await lexer.Chars.new(aiter([' foo bar\n']))
        assert (await lexer.line_comment(chars)) == '// foo bar'

        chars = await lexer.Chars.new(aiter([' foo bar']))
        assert (await lexer.line_comment(chars)) == '// foo bar'


class TestBlockComment:
    async def test_consumes_until_comment_close(self):
        chars = await lexer.Chars.new(aiter([' foo\n', 'bar\n', 'baz */\n']))
        assert (await lexer.block_comment(chars)) == '/* foo\nbar\nbaz */'

    async def test_must_have_comment_close(self):
        chars = await lexer.Chars.new(aiter([' foo']))
        with raises(lexer.InvalidCharacter):
            await lexer.block_comment(chars)
