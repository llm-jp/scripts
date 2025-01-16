import pytest

from convert import (
    split_text_by_length,
    split_text_by_bunkai,
    remove_intra_sentence_line_breaks,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", [""]),
        ("これはペンです。", ["これは", "ペンで", "す。"]),
        ("これはペンです", ["これは", "ペンで", "す"]),
    ],
)
def test_split_text_by_length(text: str, expected: list[str]) -> None:
    assert split_text_by_length(text, 3) == expected


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
    ],
)
def test_remove_intra_sentence_line_breaks(text: str, expected: str) -> None:
    assert remove_intra_sentence_line_breaks(text) == expected
