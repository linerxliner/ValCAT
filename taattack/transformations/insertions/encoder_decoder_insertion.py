from taattack.utils import detokenize, get_use_sim, model_pool, tokenize
from .insertion import Insertion


class EncoderDecoderInsertion(Insertion):
    def __init__(
        self,
        encoder_decoder='t5-base',
        max_text_length=512,
        max_candidates=50,
        num_beams=100,
        length_penalty=1e-5,
        max_filled_length=3,
        num_beam_groups=1,
        diversity_penalty=0.0,
        min_sim=0.8,
        sim_window_radius=5,
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
        self._min_sim = min_sim
        self._sim_window_radius = sim_window_radius
        self._device = device

        if self._encoder_decoder in model_pool.ENCODER_DECODER2MODEL_TOKENIZER:
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

    def _get_t5_insert_spans(self, workload, idx):
        replacement_spans = []

        masked_workload = workload.insert(idx, self._mask_token)
        if masked_workload is workload:
            return replacement_spans

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
            if t and t not in replacement_spans and t.lower() != '<unk>' and any(list(map(lambda c: c.isalnum(), t))) :
                replacement_spans.append(t)

        replacement_spans = sorted(replacement_spans[:self._max_candidates])

        return replacement_spans

    def _filter_sim_spans(self, spans, workload, idx):
        left_words = workload.words[max(0, idx - self._sim_window_radius):idx]
        right_words = workload.words[idx:min(len(workload.words), idx + self._sim_window_radius)]
        prev_text_window = detokenize(left_words + right_words)

        def check_sim_span(span):
            curr_text_window = detokenize(left_words + tokenize(span) + right_words)
            sim = get_use_sim(prev_text_window, curr_text_window)
            return sim >= self._min_sim

        return list(filter(check_sim_span, spans))

    def _get_transformed_workload_list(self, workload, idx, is_orig_idx=True):
        spans = self._get_t5_insert_spans(workload, idx)
        if spans and self._min_sim > 0:
            curr_idx = workload.orig_idx_2_curr_idx[idx].item() if is_orig_idx else idx
            spans = self._filter_sim_spans(spans, workload, curr_idx)
        transed_workload_list = [workload.insert(idx, r, is_orig_idx=is_orig_idx) for r in spans]
        transed_workload_list = list(filter(lambda w: w is not workload, transed_workload_list))

        return transed_workload_list
