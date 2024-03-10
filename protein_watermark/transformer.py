import torch
import numpy as np
from .base import AbstractReweight
from transformers import LogitsProcessor

class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(
            self,
            private_key: any,
            reweight: AbstractReweight,
            context_code_length: int,
            ignore_history=False,
    ):

        self.private_key = private_key
        self.reweight = reweight
        self.cc_length = context_code_length
        self.ignore_history = ignore_history
        self.cc_history = set()

    def get_rng_seed(self, context_code: any) -> any:
        if not self.ignore_history:
            self.cc_history.add(context_code)
        import hashlib

        m = hashlib.sha256()
        m.update(context_code)
        m.update(self.private_key)
        full_hash = m.digest()
        seed = int.from_bytes(full_hash, "big") % (2 ** 32 - 1)
        return seed

    def reset_history(self):
        self.cc_history = set()

    def _get_codes(self, context):
        batch_size = len(context)

        context_codes = [
            context[i][-self.cc_length:].tobytes() for i in range(batch_size)
        ]

        mask, seeds = zip(
            *[
                (context_code in self.cc_history, self.get_rng_seed(context_code))
                for context_code in context_codes
            ]
        )
        return mask, seeds

    def __call__(self,
                 input_ids: torch.LongTensor,
                 scores: torch.FloatTensor,
                 ):

        device = scores.device
        input_ids = input_ids.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()
        mask, seeds = self._get_codes(input_ids)

        rng = [
            np.random.default_rng(seed) for seed in seeds
        ]

        mask = np.array(mask)

        watermark_code = self.reweight.watermark_code_type.from_random(
            rng, scores.shape[1]
        )

        reweighted_scores = self.reweight.reweight(watermark_code, scores)

        if self.ignore_history:
            return torch.FloatTensor(reweighted_scores, device='cpu').to(device)
        else:
            return torch.FloatTensor(np.where(mask[:, None], scores, reweighted_scores), device='cpu').to(device)