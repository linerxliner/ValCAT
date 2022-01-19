import argparse
import json
import tensorflow as tf
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from taattack._datasets import AgNews, Imdb, Mnli, Qnli, Snli, YelpPolarity
from taattack.attackers import ValCat2022
from taattack.config import BASE_DIR, DEVICES
from taattack.victims.wrappers import HuggingFaceWrapper
from taattack.workload import Workload


ATTACKERS = {
    'valcat': ValCat2022,
}

DATASETS = {
    'ag-news': AgNews,
    'imdb': Imdb,
    'mnli': Mnli,
    'qnli': Qnli,
    'snli': Snli,
    'yelp-polarity': YelpPolarity,
}

VICTIMS = {
    'ag-news': 'textattack/bert-base-uncased-ag-news',
    'imdb': 'textattack/bert-base-uncased-imdb',
    'mnli': '/tmp/mnli',
    'qnli': '/tmp/qnli',
    'snli': '/tmp/snli',
    'yelp-polarity': 'textattack/bert-base-uncased-yelp-polarity',
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('attacker', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('--victim', type=str)
    parser.add_argument('--result', type=str)
    parser.add_argument('--devices', nargs=2, type=int)

    parser.add_argument('--dataset_slice', nargs='+', type=int)
    parser.add_argument('--attack_second_text', action='store_true')

    parser.add_argument('--max_pert_rate', type=float, default=0.4)
    parser.add_argument('--min_use_sim', type=float, default=0.8)
    parser.add_argument('--enable_trans', action='extend', nargs='+', type=int)
    parser.add_argument('--span_order', type=int, default=3)
    parser.add_argument('--encoder_decoder', type=str, default='t5-base')
    parser.add_argument('--max_candidates', type=int, default=50)

    return parser.parse_args()


def attack(attacker, dataset, result_saving_fp=None, attack_second_text=False):
    start_time = time.process_time()
    results = []

    try:
        for i, example in enumerate(dataset):
            print(f'---{i}({time.process_time() - start_time:.2f}s)---')
            if len(example) == 2:
                first_text, label = example
                second_text = None
            else:
                first_text, second_text, label = example
            if attack_second_text:
                result = attacker.attack(Workload(second_text, extra_text=first_text, label=label, text_before_extra=False))
            else:
                result = attacker.attack(Workload(first_text, extra_text=second_text, label=label))
            print(result)

            results.append(result)
    finally:
        if result_saving_fp:
            with open(result_saving_fp, 'w') as f:
                json.dump(list(map(lambda r: r.to_dict(), results)), f, indent=2, ensure_ascii=False)

    return results


if __name__ == '__main__':
    args = parse_args()

    if args.devices:
        DEVICES[0], DEVICES[1] = f'cuda:{args.devices[0]}', f'cuda:{args.devices[1]}'
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices([physical_devices[args.devices[0]], physical_devices[args.devices[1]]], 'GPU')

    attacker_kwargs = {
        'max_pert_rate': args.max_pert_rate,
        'min_use_sim': args.min_use_sim,
        'enable_trans': args.enable_trans,
        'span_order': args.span_order,
        'encoder_decoder': args.encoder_decoder,
        'max_candidates': args.max_candidates,
    }

    dataset = DATASETS[args.dataset]()
    if args.dataset_slice:
        if len(args.dataset_slice) == 1:
            dataset = dataset[args.dataset_slice[0]:]
        elif len(args.dataset_slice) == 2:
            dataset = dataset[args.dataset_slice[0]:args.dataset_slice[1]]

    model_name_or_path = VICTIMS[args.victim if args.victim else args.dataset]
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(DEVICES[1])
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model_wrapper = HuggingFaceWrapper(model, tokenizer)

    attacker = ATTACKERS[args.attacker].build(model_wrapper, **attacker_kwargs)

    result_saving_fp = BASE_DIR.joinpath(args.result) if args.result else None

    attack(attacker, dataset, result_saving_fp=args.result, attack_second_text=args.attack_second_text)
