import torch

from .wrapper import Wrapper


class PyTorchWrapper(Wrapper):
    def __init__(self, model, tokenizer):
        super(PyTorchWrapper, self).__init__()

        self._model = model
        self._tokenizer = tokenizer

        self.unk_token = self._tokenizer.unk_token

    def _forward(self, text_list):
        if len(text_list) == 1:
            text_list = text_list[0]
        ids = self._tokenizer(text_list)
        device = next(self._model.parameters()).device
        ids = torch.tensor(ids).to(device)

        with torch.no_grad():
            outputs = self._model(ids)

        return outputs.cpu().numpy()
