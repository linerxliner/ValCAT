import flair
import numpy as np
import spacy
import tensorflow_hub as hub
import torch
from flair.data import Sentence
from flair.models import SequenceTagger
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics.pairwise import cosine_similarity
from string import punctuation
from transformers import AutoTokenizer, GPT2LMHeadModel, MT5ForConditionalGeneration, T5ForConditionalGeneration

from .config import DEVICES


class ModelPool:
    ENCODER_DECODER2MODEL_TOKENIZER = {
        't5-base': 't5_base',
        't5-large': 't5_large',
        't5-v1_1-base': 't5_v1_1_base',
        'mt5-base': 'mt5_base',
    }

    def encoder_decoder2model_token(self, encoder_decoder):
        return getattr(self, self.ENCODER_DECODER2MODEL_TOKENIZER[encoder_decoder])

    @property
    def flair_pos_tagger(self):
        if not hasattr(self, '_flair_pos_tagger'):
            flair.device = torch.device(DEVICES[1])
            self._flair_pos_tagger = SequenceTagger.load('upos-fast')

        return self._flair_pos_tagger

    @property
    def gpt2(self):
        if not hasattr(self, '_gpt2_model'):
            self._gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        if not hasattr(self, '_gpt2_tokenizer'):
            self._gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)

        return self._gpt2_model, self._gpt2_tokenizer

    @property
    def mt5_base(self):
        if not hasattr(self, '_mt5_base_model'):
            self._mt5_base_model = MT5ForConditionalGeneration.from_pretrained('google/mt5-base')
        if not hasattr(self, '_mt5_base_tokenizer'):
            self._mt5_base_tokenizer = AutoTokenizer.from_pretrained('google/mt5-base', use_fast=True)

        return self._mt5_base_model, self._mt5_base_tokenizer

    @property
    def spacy_model(self):
        if not hasattr(self, '_spacy_model'):
            self._spacy_model = spacy.load('en_core_web_sm')
        return self._spacy_model

    @property
    def t5_base(self):
        if not hasattr(self, '_t5_base_model'):
            self._t5_base_model = T5ForConditionalGeneration.from_pretrained('t5-base')
        if not hasattr(self, '_t5_base_tokenizer'):
            self._t5_base_tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast=True)

        return self._t5_base_model, self._t5_base_tokenizer

    @property
    def t5_large(self):
        if not hasattr(self, '_t5_large_model'):
            self._t5_large_model = T5ForConditionalGeneration.from_pretrained('t5-large')
        if not hasattr(self, '_t5_large_tokenizer'):
            self._t5_large_tokenizer = AutoTokenizer.from_pretrained('t5-large', use_fast=True)

        return self._t5_large_model, self._t5_large_tokenizer

    @property
    def t5_v1_1_base(self):
        if not hasattr(self, '_t5_v1_1_base_model'):
            self._t5_v1_1_base_model = T5ForConditionalGeneration.from_pretrained('google/t5-v1_1-base')
        if not hasattr(self, '_t5_v1_1_base_tokenizer'):
            self._t5_v1_1_base_tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-base', use_fast=True)

        return self._t5_v1_1_base_model, self._t5_v1_1_base_tokenizer

    @property
    def treebank_word_detokenizer(self):
        if not hasattr(self, '_treebank_word_detokenizer'):
            self._treebank_word_detokenizer = TreebankWordDetokenizer()
        return self._treebank_word_detokenizer

    @property
    def use(self):
        if not hasattr(self, '_use'):
            self._use = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        return self._use


model_pool = ModelPool()


def tokenize(text):
    doc = model_pool.spacy_model(text)
    tokens = [token.text for token in doc]

    return tokens


def detokenize(tokens):
    return model_pool.treebank_word_detokenizer.detokenize(tokens)


def is_continuous(sequence):
    if len(sequence) == 0:
        return False

    for i in range(len(sequence) - 1):
        if sequence[i] + 1 != sequence[i + 1]:
            return False

    return True


def is_punctuation(c):
    return len(c) == 1 and c in punctuation


def is_one_word(text):
    return len(tokenize(text)) == 1


def get_use_sim(text1, text2):
    orig_embd, adv_embd = model_pool.use([text1, text2]).numpy()
    sim = cosine_similarity(orig_embd[np.newaxis, ...], adv_embd[np.newaxis, ...])[0, 0]
    return sim.item()


def get_lcs_len(words1, words2):
    num_words1, num_words2 = len(words1), len(words2)

    dp = np.zeros((num_words1 + 1, num_words2 + 1), dtype=int)

    for i in range(1, num_words1 + 1):
        for j in range(1, num_words2 + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])

    return dp[num_words1, num_words2].item()


def get_num_word_pert(words1, words2):
    words1, words2 = list(map(lambda w: w.lower(), words1)), list(map(lambda w: w.lower(), words2))
    return max(len(words1), len(words2)) - get_lcs_len(words1, words2)


def get_pos_list(words):
    sentence = Sentence(detokenize(words), use_tokenizer=lambda text: words)
    model_pool.flair_pos_tagger.predict(sentence)
    return [token.annotation_layers['pos'][0]._value for token in sentence.tokens]
