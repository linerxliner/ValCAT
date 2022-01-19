from taattack.attack import Attack
from taattack.config import DEVICES
from taattack.constraints import MaxWordPert, MinUseSim, Stopwords
from taattack.goals import UntargetedClassificationGoal
from taattack.search_methods import GreedySpanImportanceRanking
from taattack.transformations import EncoderDecoderInsertion, EncoderDecoderReplacement
from .attacker import Attacker


class ValCat2022(Attacker):
    @staticmethod
    def build(model_wrapper, max_pert_rate=0.4, min_use_sim=0.8, enable_trans=None, max_candidates=50, span_order=3, encoder_decoder='t5-base', **kwargs):
        trans = [
            EncoderDecoderInsertion(
                encoder_decoder=encoder_decoder,
                max_candidates=max_candidates,
                max_filled_length=3,
                device=DEVICES[1],
            ),
            EncoderDecoderReplacement(
                encoder_decoder=encoder_decoder,
                max_candidates=max_candidates,
                replace_window=1,
                max_filled_length=3,
                length_penalty=1e-6,
                device=DEVICES[1],
            ),
            EncoderDecoderReplacement(
                encoder_decoder=encoder_decoder,
                max_candidates=max_candidates,
                replace_window=2,
                max_filled_length=4,
                device=DEVICES[1],
            ),
            EncoderDecoderReplacement(
                encoder_decoder=encoder_decoder,
                max_candidates=max_candidates,
                replace_window=3,
                max_filled_length=5,
                length_penalty=1e-4,
                device=DEVICES[1],
            ),
        ]
        if enable_trans is not None:
            trans = [trans[i] for i in enable_trans]
        goal = UntargetedClassificationGoal(model_wrapper)
        constraints = [
            Stopwords(),
            MaxWordPert(max_pct=max_pert_rate),
            MinUseSim(threshold=min_use_sim),
        ]

        search_method = GreedySpanImportanceRanking(trans, goal, constraints=constraints, num_order=span_order)

        return Attack(search_method)
