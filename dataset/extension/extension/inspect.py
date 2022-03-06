import argparse
import os
import base64
from collections import Counter, defaultdict
from io import BytesIO

import webbrowser
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from jinja2 import Environment, FileSystemLoader, select_autoescape


def init_nltk():
    """Import related nltk data, download as necessary"""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger")


def load_dataset(filename: str) -> pd.Series:
    """Loads a single file of sentences into a dataframe series"""

    def line_gen(file_name):
        with open(file_name) as file_obj:
            while line := file_obj.readline():
                yield line.strip("\n")

    dataframe = pd.DataFrame(line_gen(filename), columns=["text"])
    return dataframe["text"]


def strip_outliers(series: pd.Series) -> pd.Series:
    """Filter outliers that are 3 standard deviations from the mean"""
    return series[((series - series.mean()) / series.std()).abs() < 3]


class Analyzer:
    def __init__(self, dataset: pd.Series, skip_nlp: bool = False):
        self.dataset = dataset
        if not skip_nlp:
            init_nltk()
            self.stop = set(stopwords.words("english"))
            self.nlp = spacy.load(
                "en_core_web_sm",
                exclude=[
                    "tok2vec",
                    "tagger",
                    "parser",
                    "senter",
                    "attribute_ruler",
                    "lemmatizer",
                ],
            )
            self.init_nlp()

    def init_nlp(self):

        self.entities = defaultdict(Counter)
        self.pos_tags = defaultdict(Counter)

        def apply_nlp(text):
            doc = self.nlp(text)
            for entity in doc.ents:
                self.entities[entity.label_].update([entity.text])

            for text, pos in nltk.pos_tag(word_tokenize(text)):
                self.pos_tags[pos].update([text])

            # Spacy POS taggin does recognize any tags for some reason
            # for token in doc:
            #     self.pos_tags[token.pos_].update([token.text])

        self.dataset.apply(lambda text: apply_nlp(text))

        entity_counts = {}
        for entity, counts in self.entities.items():
            # Python 3.10 has a nice total method to get sum of all values
            entity_counts[entity] = sum(counts.values())
        self.entity_counts = Counter(entity_counts)

        pos_tag_counts = {}
        for pos_tag, counts in self.pos_tags.items():
            # Python 3.10 has a nice total method to get sum of all values
            pos_tag_counts[pos_tag] = sum(counts.values())
        self.pos_tag_counts = Counter(pos_tag_counts)

    @property
    def char_lengths(self):
        return self.dataset.str.len()

    @property
    def word_count(self):
        return self.dataset.apply(self._word_filter).str.split().map(lambda x: len(x))

    @staticmethod
    def _word_filter(text):
        return text.replace(".", "").replace("?", "").replace("!", "").replace(",", "")

    def _word_gen(self):
        for sent in self.dataset.apply(self._word_filter).str.split().array:
            for word in sent:
                yield word

    @property
    def words(self):
        return (word for word in self._word_gen())

    @property
    def unique_words(self):
        word_set = set()
        for word in self.words:
            word_set |= set([word.lower()])
        return word_set

    @property
    def stopwords(self):
        return (word.lower() for word in self.words if word.lower() in self.stop)

    def top_stopwords(self, top: int = 10):
        return Counter(self.stopwords).most_common(top)

    @property
    def non_stopwords(self):
        return (word.lower() for word in self.words if word.lower() not in self.stop)

    def top_non_stopwords(self, top: int = 10):
        return Counter(self.non_stopwords).most_common(top)

    def top_ngram(self, n_value: int, limit: int = 10):
        vec = CountVectorizer(ngram_range=(n_value, n_value)).fit(self.dataset)
        bag_of_words = vec.transform(self.dataset)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [
            (word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()
        ]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:limit]

    def top_name_entities(self, top: int = 10):
        return self.entity_counts.most_common(top)

    def top_parts_of_speech(self, top: int = 10):
        return self.pos_tag_counts.most_common(top)


class Renderer:
    def __init__(self, data: Analyzer):
        self.data = data
        self.figure = plt.figure()
        self.jinja_env = Environment(
            loader=FileSystemLoader(
                f"{os.getcwd()}/dataset/extension/templates"
            ),
            autoescape=select_autoescape(),
        )

    def render_data(self):
        return {
            "length_plots": [
                self.char_length_hist(),
                self.word_count_hist(),
            ],
            "ngram_plots": [
                self.plot_top_ngrams_barchart(2),
                self.plot_top_ngrams_barchart(3),
            ],
            "stopwords_plots": [
                self.plot_top_stopwords_barchart(),
                self.plot_top_non_stopwords_barchart(),
            ],
            "named_entity_plots": [
                self.plot_named_entity_barchart(),
                self.plot_named_entity_words_barchart(1),
                self.plot_named_entity_words_barchart(2),
                self.plot_named_entity_words_barchart(3),
            ],
            "parts_of_speech_plots": [
                self.plot_parts_of_speach_barchart(),
                self.plot_named_parts_of_speach_barchart(1),
                self.plot_named_parts_of_speach_barchart(2),
                self.plot_named_parts_of_speach_barchart(3),
            ],
            "sentence_count": f"{self.data.dataset.count():,}",
            "unique_words": f"{len(self.data.unique_words):,}",
        }

    def render_to_file(self, file_path: str):
        template = self.jinja_env.get_template("template.html")
        with open(file_path, "w") as fobj:
            fobj.write(template.render(**self.render_data()))

    def figure_to_image(self) -> str:
        tmpfile = BytesIO()
        self.figure.savefig(tmpfile, format="png")
        encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
        self.figure.clear()
        return f"data:image/png;base64,{encoded}"

    def add_caption(self, text: str):
        self.ax_list = self.figure.axes
        self.ax_list[0].set_xlabel(f" Label \n {text}")

    def _count_hist(self, dataset: pd.Series, title: str, xlabel: str):
        dataset.hist()
        plt.xlabel(xlabel)
        plt.title(title)
        plt.grid(visible=None)
        # Some magic number here to get the label spacing right
        _, max_ylim = plt.ylim()
        # Average value line
        plt.axvline(dataset.mean(), color="k", linestyle="dashed", linewidth=1)
        plt.text(dataset.mean() * 1.1, max_ylim * 0.9, f"Average: {dataset.mean():.2f}")
        # Minimum value line
        plt.axvline(dataset.min(), color="k", linestyle="dashed", linewidth=1)
        plt.text(dataset.min() * 1.1, max_ylim * 0.9, f"Least: {dataset.min()}")
        # Maximum value line
        plt.axvline(dataset.max(), color="k", linestyle="dashed", linewidth=1)
        plt.text(dataset.max() * 0.85, max_ylim * 0.9, f"Most: {dataset.max()}")
        return self.figure_to_image()

    def char_length_hist(self) -> str:
        dataset = self.data.char_lengths
        return self._count_hist(
            dataset, "Character Count per Sentence", "Character Count"
        )

    def word_count_hist(self) -> str:
        dataset = self.data.dataset.str.split().map(lambda x: len(x))
        return self._count_hist(dataset, "Word Count per Sentence", "Word Count")

    def plot_top_stopwords_barchart(self):
        x, y = map(list, zip(*self.data.top_stopwords()))
        sns.barplot(x=y, y=x)
        plt.title("Top Stopwords")
        plt.tight_layout()
        return self.figure_to_image()

    def plot_top_non_stopwords_barchart(self):
        x, y = map(list, zip(*self.data.top_non_stopwords()))
        sns.barplot(x=y, y=x)
        plt.title(f"Top Non-Stopwords")
        plt.tight_layout()
        return self.figure_to_image()

    def plot_top_ngrams_barchart(self, num: int):
        top_ngrams = self.data.top_ngram(n_value=num)
        x, y = map(list, zip(*top_ngrams))
        sns.barplot(x=y, y=x)
        prefix = {
            2: "Bi",
            3: "Tri",
        }
        plt.title(f"Top {prefix[num]}grams")
        plt.tight_layout()
        return self.figure_to_image()

    def plot_named_entity_barchart(self):
        x, y = map(list, zip(*self.data.top_name_entities()))
        sns.barplot(x=y, y=x)
        plt.title(f"Top Named Entities")
        plt.tight_layout()
        return self.figure_to_image()

    def plot_named_entity_words_barchart(self, num: int = 1):
        top_entity = self.data.top_name_entities(num)[num - 1][0]
        x, y = map(list, zip(*self.data.entities[top_entity].most_common(10)))
        sns.barplot(x=y, y=x)
        plt.title(top_entity)
        plt.tight_layout()
        return self.figure_to_image()

    def plot_parts_of_speach_barchart(self):
        x, y = map(list, zip(*self.data.top_parts_of_speech()))
        sns.barplot(x=y, y=x)
        plt.title(f"Top Parts of Speech")
        plt.tight_layout()
        return self.figure_to_image()

    def plot_named_parts_of_speach_barchart(self, num: int = 1):
        top_pos = self.data.top_parts_of_speech(num)[num - 1][0]
        x, y = map(list, zip(*self.data.pos_tags[top_pos].most_common(10)))
        sns.barplot(x=y, y=x)
        plt.title(top_pos)
        plt.tight_layout()
        return self.figure_to_image()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="inspect",
        description="""
            Script to perform operations on a processed dataset.
            By default this will generate a short report on various information on the datset.
            With the `words_export` flag this will create a txt file of unique words in the dataset.
        """,
    )
    parser.add_argument(
        "--source_text",
        required=True,
        help="Source text document of sentences separated by newlines",
    )
    parser.add_argument(
        "--output_path", required=True, help="Filepath for the output file"
    )
    parser.add_argument(
        "--words-export",
        action="store_true",
        help="Flag to export unique words to the given output path",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Flag to open corpus report in the browser",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.source_text)
    output_path = args.output_path

    if args.words_export:
        processed = Analyzer(dataset, skip_nlp=True)
        with open(output_path, "a") as file_obj:
            for word in processed.unique_words:
                file_obj.write(f"{word}\n")
    else:
        processed = Analyzer(dataset)
        renderer = Renderer(processed)

        if os.path.exists(output_path):
            os.remove(output_path)

        renderer.render_to_file(output_path)

        if args.interactive:
            webbrowser.open(f"file://{os.path.abspath(output_path)}")
