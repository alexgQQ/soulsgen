"""
Extension module for Expanda.
This is meant to handle XML files that are unpacked from Dark Souls FMG files.
Essentially just a XML parser with specific methods to clean and format
odd or uneeded text entries.
"""

from typing import Dict, Any
import xml.etree.ElementTree as ET
import nltk  # pylint: disable=E0401
from .utils import (
    _clear_between_labels,
    _clear_parenthesis,
    _clear_brackets,
    _clear_slashes_ellipses,
    _clear_formatting,
    _split_sentences,
)


def clean_text_entry(entry: str):
    """
    Properly format a ds description entry
    Removes metadata tags and formatting so the more pure
    description of action and lore remain in a list of sentences
    """
    entry = _clear_parenthesis(entry)
    entry = _clear_brackets(entry)
    entry = _clear_slashes_ellipses(entry)
    entry = _clear_between_labels(entry, "Weapon type:")
    entry = _clear_between_labels(entry, "Attack type:")
    entry = _clear_between_labels(entry, "Skill:")
    # Removes newlines, should run last as some prior regex relies on it
    entry = _clear_formatting(entry)
    for i in range(1, 16):
        entry.replace(f"+{i}", "")

    return entry


def traverse_text(entries_elem):
    """
    Filter out nulled or duplicate elements
    The `any` check is fine but will scale poorly if data is too large.
    """
    entries = []
    to_ignore = (
        "%null%",
        "no_text",
        "no text",
        "none text",
        " ",
        "  ",
        "(dummyText)"
    )

    in_check = (
        "#c", "&lt", "&gt", "multiplayer", "session", "NAT", "PSN", "network", "service",
        "DARK SOULS", "SEKIRO", "amiibo", "settings", "menu"
    )

    delimeters = (
        ".", "?", "!", ",", "..."
    )
    for text_elem in entries_elem:
        text = text_elem.text
        if any((text == ignore for ignore in to_ignore)):
            continue
        if any((ignore in text for ignore in in_check)):
            continue
        if not any((text.endswith(delim) for delim in delimeters)):
            continue
        text = clean_text_entry(text)
        if len(text) < 5 or len(text.split()) < 1:
            continue
        entries.append(text)
    # Strip duplicates, not sure if this is something proper
    # but some of the documents have lots of duplicated statements
    entries = set(entries)
    return list(entries)


def parse_xml(xml_file):
    """
    Parse a ds xml file from fmg long_ description formats
    and retrieve clean item descriptions as individual sentences
    """
    tree = ET.parse(xml_file)
    root_elem = tree.getroot()
    return traverse_text(root_elem[3])


def handle_ds_export(
    input_file: str,
    output_file: str,
    temporary: str,  # pylint: disable=W0613
    args: Dict[str, Any],  # pylint: disable=W0613
):
    """
    Main entrypoint for the custom Expanda extension
    https://expanda.readthedocs.io/en/latest/expanda.extension.html
    """

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    nltk.data.find("tokenizers/punkt")
    tokenize_sentence = nltk.tokenize.sent_tokenize

    # Get all the entries as a single string
    # so the whole doc can be tokenized properly
    sentences = parse_xml(input_file)
    text = " ".join(sentences)

    with open(output_file, "w", encoding="utf-8") as dst:
        for tk_sentence in tokenize_sentence(text):
            dst.write(tk_sentence + "\n")


__extension__ = {
    "name": "soulsgen-xml",
    "version": "1.0",
    "description": "A parser extension for Dark Souls game files. \
        Handles the text literal XML files from FMG files.",
    "author": "Alex Grand",
    "main": handle_ds_export,
    "arguments": {},
}
