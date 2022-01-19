class Dataset:
    _data = None
    _first_text_col = 'text'
    _second_text_col = None
    _label_col = 'label'

    def __init__(self):
        self._idx = 0
        if self._data is None:
            raise Exception('Dataset is not loaded')

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self._data):
            raise StopIteration
        else:
            item = self._data.iloc[self._idx]
            self._idx += 1

            if self._second_text_col:
                return item[self._first_text_col], item[self._second_text_col], int(item[self._label_col])
            else:
                return item[self._first_text_col], int(item[self._label_col])

    def __getitem__(self, item):
        if isinstance(item, int):
            item = self._data.iloc[item]

            if self._second_text_col:
                return item[self._first_text_col], item[self._second_text_col], int(item[self._label_col])
            else:
                return item[self._first_text_col], int(item[self._label_col])
        elif isinstance(item, slice):
            start = item.start if item.start else 0
            stop = item.stop if item.stop else len(self._data)
            step = item.step if item.step else 1

            items = self._data.iloc[start:stop:step]

            if self._second_text_col:
                return [(item[self._first_text_col], item[self._second_text_col], int(item[self._label_col])) for _, item in items.iterrows()]
            else:
                return [(item[self._first_text_col], int(item[self._label_col])) for _, item in items.iterrows()]
        else:
            raise KeyError

    def __str__(self):
        return str(self._data)