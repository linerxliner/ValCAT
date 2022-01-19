import csv
import pandas as pd

from taattack._datasets.dataset import Dataset
from taattack.config import BASE_DIR


class Snli(Dataset):
    def __init__(self):
        self._data = pd.read_csv(
            BASE_DIR.joinpath('_datasets/snli/snli.tsv'),
            sep='\t',
            usecols=['label', 'premise', 'hypothesis'],
            quoting=csv.QUOTE_NONE,
        )
        self._first_text_col = 'premise'
        self._second_text_col = 'hypothesis'

        super().__init__()
