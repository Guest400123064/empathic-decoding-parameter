# %%
from typing import List, Dict, Any, Callable, Union
from itertools import combinations

import re

from sklearn.metrics.pairwise import cosine_similarity

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from empath import Empath
from sentence_transformers import SentenceTransformer


class SentenceBERTDiversity:
    """Use SentenceBERT to compute the average pairwise cosine similarity 
        between a set of utterances. This is used to measure the 
        diversity of a set of responses. The set can be generated from
        multiple-runs or a series of responses from a single 
        dialogue session."""

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the SentenceBERT diversity metric.
        """
        self._model = SentenceTransformer(model)

    @property
    def name(self) -> str:
        """Name of the metric.
        """
        return "sentencebert_diversity"

    def __repr__(self) -> str:
        """String representation of the SentenceBERT diversity metric.
        """
        return f"SentenceBERTDiversity({self._model})"

    def __call__(self, utterances: Union[List[str], List[List[str]]]) -> List[float]:
        """Compute the SentenceBERT diversity metric for a batch of utterance sets.
        """
        if isinstance(utterances[0], str):
            utterances = [utterances]

        return [self._analyze(s) for s in utterances]

    def _analyze(self, utterances: List[str]) -> float:
        """Compute the SentenceBERT diversity metric for a single utterance set.
        """
        pairs = list(combinations(self._model.encode(utterances), 2))
        return 1 - cosine_similarity(*zip(*pairs)).mean()


class Length:
    """Simply the length of an utterance."""

    @property
    def name(self) -> str:
        """Name of the metric.
        """
        return "length"

    def __repr__(self) -> str:
        """String representation of the length metric.
        """
        return f"Length()"

    def __call__(self, utterances: Union[str, List[str]]) -> List[float]:
        """Compute the length metric for a batch of utterances.
        """
        if isinstance(utterances, str):
            utterances = [utterances]

        return [self._analyze(utterance) for utterance in utterances]

    def _analyze(self, utterance: str) -> float:
        """Compute the length metric for a single utterance.
        """
        return len(utterance.split())


class EmpathNegativity:
    """Empath negativity metric."""

    def __init__(self, 
                 categories: List[str] = None, 
                 normalize: bool = True) -> None:
        """Initialize the Empath polarity metric.
        """
        self._empath     = Empath()
        self._normalize  = normalize
        self._categories = categories or ["negative_emotion",
                                          "aggression",
                                          "hate"]

        self._stopwords = set(stopwords.words("english"))
        self._lemmatize = WordNetLemmatizer().lemmatize

    @property
    def name(self) -> str:
        """Name of the metric.
        """
        return "empath_polarity"

    def __repr__(self) -> str:
        """String representation of the Empath polarity metric.
        """
        return f"EmpathPolarity({self._categories}, {self._normalize})"

    def __call__(self, utterances: Union[str, List[str]]) -> List[float]:
        """Compute the Empath polarity metric for a batch of utterances.
        """
        if isinstance(utterances, str):
            utterances = [utterances]

        return [self._analyze(utterance) for utterance in utterances]

    def _analyze(self, utterance: str) -> float:
        """Compute the Empath polarity metric for a single utterance.
        """
        scores = self._empath.analyze(self._preprocess(utterance),
                                      categories=self._categories, 
                                      normalize=self._normalize)
        return sum(scores.values()) / len(scores)

    def _preprocess(self, utterance: str) -> str:
        """Preprocess an utterance. Remove stopwords, punctuations, 
            and lemmatize to base form.
        """
        utterance = re.sub(r"[^\w\s\d]", "", utterance)

        # Remove stopwords and lemmatize
        utterance = " ".join([self._lemmatize(word) for word in word_tokenize(utterance) 
                              if word not in self._stopwords])
        return utterance.lower()

# %%
