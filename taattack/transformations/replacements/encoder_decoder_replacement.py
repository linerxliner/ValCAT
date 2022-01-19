from taattack.utils import detokenize, get_use_sim, model_pool, tokenize
from .replacement import Replacement


class EncoderDecoderReplacement(Replacement):
    def __init__(
        self,
        encoder_decoder='t5-base',
        max_text_length=512,
        max_candidates=50,
        num_beams=100,
        length_penalty=1e-4,
        max_filled_length=3,
        num_beam_groups=1,
        diversity_penalty=0.0,
        sim_limit=0.8,
        sim_window_radius=5,
        replace_window=1,
        window_sliding=False,
        device='cpu',
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._encoder_decoder = encoder_decoder
        self._max_text_length = max_text_length
        self._max_candidates = max_candidates
        self._num_beams=num_beams
        self._length_penalty = length_penalty
        self._max_filled_length = max_filled_length
        self._num_beam_groups = num_beam_groups
        self._diversity_penalty = diversity_penalty
        self._sim_limit = sim_limit
        self._sim_window_radius = sim_window_radius
        self._window_sliding = window_sliding
        self._device = device
        self.replace_window = replace_window

        if encoder_decoder in model_pool.ENCODER_DECODER2MODEL_TOKENIZER:
            self._model, self._tokenizer = model_pool.encoder_decoder2model_token(encoder_decoder)
            self._mask_token = '<extra_id_0>'
            self._extra_eos_token = '<extra_id_1>'
        else:
            raise Exception('Unknown encoder-decoder model')
        self._model.to(self._device)
        self._model.eval()

    def _encode_text(self, text):
        encoding = self._tokenizer(
            text,
            add_special_tokens=True,
            max_length=self._max_text_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        return encoding.to(self._device)

    def _get_replacement_words(self, workload, idx, max_candidates=50, num_replaced=1, is_orig_idx=True):
        replacement_spans = []

        masked_workload = workload.replace(idx, self._mask_token, num_replaced=num_replaced, is_orig_idx=is_orig_idx)
        if masked_workload is workload:
            return replacement_spans
        prev_span = detokenize(masked_workload.modified_log[-1]['old_words'])

        inputs = self._encode_text(masked_workload.full_text)
        outputs = self._model.generate(
            **inputs,
            eos_token_id=self._tokenizer.convert_tokens_to_ids(self._extra_eos_token),
            max_length=2 + self._max_filled_length,
            early_stopping=True,
            num_beams=self._num_beams,
            length_penalty=self._length_penalty,
            no_repeat_ngram_size=2,
            num_return_sequences=self._num_beams,
            num_beam_groups=self._num_beam_groups,
            diversity_penalty=self._diversity_penalty,
        )
        texts = self._tokenizer.batch_decode(
            outputs[:, 2:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for t in texts:
            if self._extra_eos_token in t:
                t = t[:t.index(self._extra_eos_token)]
            if '</s>' in t:
                t = t[:t.index('</s>')]
            if t and t not in replacement_spans and t.lower() not in [prev_span.lower(), '<unk>'] and any(list(map(lambda c: c.isalnum(), t))):
                replacement_spans.append(t)

        replacement_spans = sorted(replacement_spans[:max_candidates])

        return replacement_spans

    def _filter_sim_spans(self, spans, workload, left, is_orig_idx=True):
        right = left + self.replace_window - 1
        if is_orig_idx:
            left, right = workload.orig_idx_2_curr_idx[left], workload.orig_idx_2_curr_idx[right]
        left_words = workload.words[max(0, left - self._sim_window_radius):left]
        right_words = workload.words[right + 1:min(workload.num_words, right + 1 + self._sim_window_radius)]
        prev_text_window = detokenize(left_words + workload.words[left:right + 1] + right_words)

        def check_sim_span(span):
            curr_text_window = detokenize(left_words + tokenize(span) + right_words)
            sim = get_use_sim(prev_text_window, curr_text_window)
            return sim >= self._sim_limit

        return list(filter(check_sim_span, spans))

    def _get_transformed_workload_list(self, workload, idx, is_orig_idx=True):
        transformed_workload_list = []

        num_words = workload.orig.num_words if is_orig_idx else workload.num_words

        min_left = max(0, idx - self.replace_window + 1)
        if self._window_sliding:
            max_left = min(idx, num_words - self.replace_window)
        else:
            max_left = idx - self.replace_window + 1

        if max_left < min_left:
            return transformed_workload_list
        candidates_per_win = self._max_candidates // (max_left - min_left + 1)

        for left in range(min_left, max_left + 1):
            spans = self._get_replacement_words(workload, left, max_candidates=candidates_per_win, num_replaced=self.replace_window, is_orig_idx=is_orig_idx)
            if spans and self._sim_limit > 0:
                spans = self._filter_sim_spans(spans, workload, left, is_orig_idx=is_orig_idx)
            workload_list = [workload.replace(left, r, num_replaced=self.replace_window, is_orig_idx=is_orig_idx) for r in spans]
            workload_list = list(filter(lambda w: w is not workload, workload_list))

            transformed_workload_list.extend(workload_list)

        return transformed_workload_list
