import pytest

from convert import (
    split_text_by_newline,
    split_text_by_bunkai,
    remove_intra_sentence_line_breaks,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello World\n", ["Hello ", "World\n"]),
        ("Hello\nWorld", ["Hello\nWorld"]),
        ("Hello\nWorld\n", ["Hello\nWorld\n"]),
        ("Hello\nWorld\nHello\nWorld\n", ["Hello\nWorld\nHello\nWorld\n"]),
        (
            "CONTEXT|Hello\nWorld\nHello\nWorld|CONTEXT",
            ["CONTEXT|", "Hello\nWorld\nHello\nWorld", "|CONTEXT"],
        ),
    ],
)
def test_split_text_by_newline(text: str, expected: list[str]) -> None:
    assert split_text_by_newline(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "こういう\n日本語の文章は\nよくあります",
            ["こういう\n日本語の文章は\nよくあります"],
        ),
        (
            "改行が文区切りです\nこういう日本語の文章はよくあります",
            ["改行が文区切りです\n", "こういう日本語の文章はよくあります"],
        ),
        (
            "改行が文区切り\nです\nこういう日本語\nの文章はよくあ\nります\n",
            ["改行が文区切り\nです\n", "こういう日本語\nの文章はよくあ\nります\n"],
        ),
        (
            "\n",
            ["\n"],
        ),
    ],
)
def test_split_text_by_bunkai(text: str, expected: list[str]) -> None:
    assert split_text_by_bunkai(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "こういう\n日本語の文章は\nよくあります",
            "こういう日本語の文章はよくあります",
        ),
        (
            "\n\nこういう\n日本語の文章は\nよくあります\n\n",
            "\n\nこういう日本語の文章はよくあります\n\n",
        ),
        (
            "\nこういう\n日本語の文章は\nよくあります\n",
            "\nこういう日本語の文章はよくあります\n",
        ),
        (
            "\n\n\nこういう\n日本語の文章は\nよくあります\n",
            "\n\n\nこういう日本語の文章はよくあります\n",
        ),
        (
            "\n",
            "\n",
        ),
        (
            "\n\n",
            "\n\n",
        ),
    ],
)
def test_remove_intra_sentence_line_breaks(text: str, expected: str) -> None:
    assert remove_intra_sentence_line_breaks(text) == expected
