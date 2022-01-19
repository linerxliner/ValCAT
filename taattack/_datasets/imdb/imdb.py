import csv
import pandas as pd

from taattack._datasets.dataset import Dataset
from taattack.config import BASE_DIR


class Imdb(Dataset):
    def __init__(self):
        self._data = pd.read_csv(
            BASE_DIR.joinpath('_datasets/imdb/imdb.tsv'),
            sep='\t',
            usecols=['label', 'text'],
            quoting=csv.QUOTE_NONE,
        )
        self._first_text_col = 'text'

        super().__init__()
