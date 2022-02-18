from extension import __version__
from extension.utils import (
    _clear_between_labels,
    _clear_brackets,
    _clear_formatting,
    _clear_parenthesis,
    _clear_slashes_ellipses,
    _split_sentences,
    _split_words,
)


test_xml = """<?xml version="1.0" encoding="utf-8"?>
<fmg>
<compression>None</compression>
<version>DarkSouls3</version>
<bigendian>False</bigendian>
<entries>
<text id="1000001">It is called Lothric,</text>
<text id="1000002">where the transitory lands of the Lords of Cinder converge.</text>
<text id="113">Online play item.
Invade world of player in Book of the Guilty.
Subdue player to acquire Souvenir of Reprisal.
(Only Covenanter can use the item)</text>
<text id="1107200">Weapon type: Halberd
Attack type: Slash

Halberd with a  large blade.
Scythe adjusted for battle.

Designed especially for slicing. However,
one false swing leaves one wide open.
</text>
</entries>
</fmg>
"""


def test_version():
    assert __version__ == "1.0.0"


def test_clear_between_labels():
    input_text = "Only this should remain Label: this should all be gone!"
    expected_text = "Only this should remain !"
    result = _clear_between_labels(input_text, "Label:", label_end="!")
    assert result == expected_text

    input_text = "A more realistic example Weapon Type: Slash\n"
    expected_text = "A more realistic example \n"
    result = _clear_between_labels(input_text, "Weapon Type:")
    assert result == expected_text


def test_clear_brackets():
    input_text = "Data between <brackets> should be removed"
    expected_text = "Data between  should be removed"
    result = _clear_brackets(input_text)
    assert result == expected_text


def test_clear_formatting():
    input_text = "\nNewlines should\nbecome spaces. \tTabs removed.   Multiple  and excess spaces condensed. "
    expected_text = "Newlines should become spaces. Tabs removed. Multiple and excess spaces condensed."
    result = _clear_formatting(input_text)
    assert result == expected_text


def test_clear_parenthesis():
    input_text = "Anything between (parenthesis) should be removed"
    expected_text = "Anything between  should be removed"
    result = _clear_parenthesis(input_text)
    assert result == expected_text


def test_clear_slashes_ellipses():
    input_text = "Slashes/and\\ellipses...should/become...spaces\\"
    expected_text = "Slashes and ellipses should become spaces "
    result = _clear_slashes_ellipses(input_text)
    assert result == expected_text


def test_split_sentences():
    input_text = (
        "These sentences. Should be split into an array. This counts! Did it work?."
    )
    expected_texts = [
        "These sentences.",
        "Should be split into an array.",
        "This counts! Did it work?",
    ]
    results = _split_sentences(input_text)
    for expected, result in zip(expected_texts, results):
        assert result == expected


def test_split_words():
    input_text = " Hey, all these words! Should be split. Did they? "
    expected_texts = [
        "Hey",
        "all",
        "these",
        "words",
        "Should",
        "be",
        "split",
        "Did",
        "they",
    ]
    results = _split_words(input_text)
    for expected, result in zip(expected_texts, results):
        assert result == expected
