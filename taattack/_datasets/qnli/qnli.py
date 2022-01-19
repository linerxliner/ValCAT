import csv
import pandas as pd

from taattack._datasets.dataset import Dataset
from taattack.config import BASE_DIR


class Qnli(Dataset):
    def __init__(self):
        self._data = pd.read_csv(
            BASE_DIR.joinpath('_datasets/qnli/qnli.tsv'),
            sep='\t',
            usecols=['label', 'question', 'sentence'],
            quoting=csv.QUOTE_NONE,
        )
        self._first_text_col = 'question'
        self._second_text_col = 'sentence'

        super().__init__()
