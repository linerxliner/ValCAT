import nltk
from string import punctuation

from .pre_constraint import PreConstraint


class Stopwords(PreConstraint):
    def __init__(self, stopwords=None, lang='english'):
        if stopwords is not None:
            self._stopwords = set(stopwords)
        else:
            self._stopwords = set(nltk.corpus.stopwords.words(lang) + list(punctuation))

    def filter(self, indices, workload):
        filtered_ids = []

        for idx in indices:
            if isinstance(idx, int):
                if workload.words[idx].lower() not in self._stopwords:
                    filtered_ids.append(idx)
            elif isinstance(idx, tuple) and len(idx) == 2:
                if idx[1] > 1 or workload.words[idx[0]].lower() not in self._stopwords:
                    filtered_ids.append(idx)

        return filtered_ids
