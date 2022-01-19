import numpy as np
from copy import deepcopy

from taattack.utils import detokenize, is_continuous, get_num_word_pert, tokenize


class Workload:
    def __init__(self, text, extra_text=None, label=None, prev=None, orig=None, text_before_extra=True):
        self.extra_text = extra_text
        self.label = label
        self.prev = prev
        self.orig = self if orig is None else orig
        self.text_before_extra = text_before_extra

        self.queries = 0
        self.words = tokenize(text)
        self.orig_idx_2_curr_idx = np.arange(self.num_words + 1)
        self.modified_log = []

        self.failed_reason = None
        self.probs = None
        self.sim = None
        self.num_word_pert = 0

    def __str__(self):
        return self.full_text

    @property
    def text(self):
        return detokenize(self.words)

    @property
    def full_text(self):
        if self.extra_text is None:
            return self.text
        else:
            if self.text_before_extra:
                return ' '.join([self.text, self.extra_text])
            else:
                return ' '.join([self.extra_text, self.text])

    @property
    def num_words(self):
        return len(self.words)

    def to_dict(self):
        d = {
            'text': self.text,
            'extra_text': self.extra_text if self.extra_text else '',
            'label': self.label,
            'words': self.words,
            'orig_idx_2_curr_idx': self.orig_idx_2_curr_idx.tolist(),
            'modified_log': self.modified_log,
            'queries': self.queries,
            'num_word_pert': self.num_word_pert,
        }
        if self.sim is not None:
            d['sim'] = self.sim
        if self.probs is not None:
            d['probs'] = self.probs.tolist()

        return d

    def replace(self, idx, text, num_replaced=1, is_orig_idx=True):
        indices = list(range(idx, idx + num_replaced))
        if indices[-1] >= (self.orig.num_words if is_orig_idx else self.num_words):
            return self

        curr_indices = list(map(lambda i: self.orig_idx_2_curr_idx[i].item(), indices)) if is_orig_idx else indices
        if -1 in curr_indices or not is_continuous(curr_indices) or curr_indices[-1] >= len(self.words):
            return self
        new_words = tokenize(text)
        num_new_word = len(new_words)

        copy = deepcopy(self)

        old_words = self.words[curr_indices[0]:curr_indices[-1] + 1]
        copy.words[curr_indices[0]:curr_indices[-1] + 1] = new_words
        copy.orig_idx_2_curr_idx[idx:idx + num_replaced] = -1
        incred_orig_idx_2_curr_idx = copy.orig_idx_2_curr_idx[idx + num_replaced:]
        incred_orig_idx_2_curr_idx[incred_orig_idx_2_curr_idx != -1] += num_new_word - num_replaced

        copy.modified_log.append({
            'type': 'replace',
            'old_words': old_words,
            'new_words': new_words,
            'replaced_indices': curr_indices,
            'num_modified_indices': (self.modified_log[-1]['num_modified_indices'] if self.modified_log else 0) + len(curr_indices)
        })
        copy.num_word_pert += get_num_word_pert(old_words, new_words)
        copy.prev = self
        copy.orig = self.orig

        return copy

    def insert(self, idx, text, is_orig_idx=True, allow_repeat=False):
        curr_idx = self.orig_idx_2_curr_idx[idx].item() if is_orig_idx else idx
        if curr_idx == -1 or (is_orig_idx and not allow_repeat and ((idx == 0 and curr_idx > 0) or (idx > 0 and (self.orig_idx_2_curr_idx[idx - 1].item() + 1) < curr_idx))):
            return self
        new_words = tokenize(text)
        num_new_word = len(new_words)

        copy = deepcopy(self)

        copy.words = copy.words[:curr_idx] + new_words + copy.words[curr_idx:]
        incred_orig_idx_2_curr_idx = copy.orig_idx_2_curr_idx[idx:]
        incred_orig_idx_2_curr_idx[incred_orig_idx_2_curr_idx != -1] += num_new_word

        copy.modified_log.append({
            'type': 'insert',
            'new_words': new_words,
            'insert_before_idx': curr_idx,
            'num_modified_indices': (self.modified_log[-1]['num_modified_indices'] if self.modified_log else 0) + 1
        })
        copy.num_word_pert += len(new_words)
        copy.prev = self
        copy.orig = self.orig

        return copy
