"""Save all unique words from a text file of sentences separated by newlines"""
import os
import argparse
from .utils import _split_words


def line_gen(file_name):
    """Read file line by line as a generator"""
    with open(file_name) as file_obj:
        while line := file_obj.readline():
            yield line.strip("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="words",
        description="Script to gather unique words from a set of sentences",
    )
    parser.add_argument(
        "--source_text",
        required=True,
        help="Source text document of sentences separated by newlines",
    )
    parser.add_argument(
        "--output_path", required=True, help="Filepath for the output file"
    )
    args = parser.parse_args()

    if os.path.exists(args.output_path):
        os.remove(args.output_path)

    word_set = set()
    for words in line_gen(args.source_text):
        words = _split_words(words)
        for word in words:
            word_set.add(word)

    with open(args.output_path, "a") as file_obj:
        for word in word_set:
            file_obj.write(f"{word}\n")
