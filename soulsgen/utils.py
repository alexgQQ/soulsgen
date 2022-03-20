""" Specific utils for formatting strings """

import re
from typing import List


def _clear_between_labels(data: str, label_start: str, label_end: str = "\n") -> str:
    """Remove text between two labels, remove the beginning label but leave the ending label"""
    return re.sub(f"(?<={label_start}).*(?={label_end})", "", data).replace(
        label_start, ""
    )


def _clear_parenthesis(data: str) -> str:
    """Remove parenthesis and text between, (like this)"""
    return re.sub(r"\([^()]*\)", "", data)


def _clear_brackets(data: str) -> str:
    """Remove brackets and text between, <like this>"""
    return re.sub(r"\<[^()]*\>", "", data)


def _clear_slashes_ellipses(data: str) -> str:
    """Replace forward slash, double backslash and ellipses with a single space"""
    return data.replace("/", " ").replace("\\", " ").replace("...", " ")


def _clear_formatting(data: str) -> str:
    """Remove tabs, replace newlines with a space and condense multiple spaces to a single space"""
    data = (
        data.replace("\n", " ")
        .replace("\t", "")
        .replace("&s", "")
        .replace("&d", "")
        .replace(":", "")
    )
    return re.sub(" +", " ", data).strip()


def _split_sentences(data: str) -> List[str]:
    """Split a string into sentences and into an array, ignores ? or !"""
    sentences = data.split(".")
    sentences = list(filter(lambda x: len(x) > 1, sentences))
    sentences = [
        f"{sentence}{'' if sentence.endswith('?') or sentence.endswith('!') else '.'}".strip()
        for sentence in filter(bool, sentences)
    ]
    return sentences


def _split_words(data: str) -> List[str]:
    """Split a sentence into separate words"""
    return (
        data.replace("?", "")
        .replace(",", "")
        .replace("!", "")
        .replace(".", "")
        .replace('"', "")
        .strip()
        .split()
    )
