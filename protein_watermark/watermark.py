import numpy as np
from .base import AbstractReweight


"""
Codes below are modified from unbiased watermark, 
GitHub repo: https://github.com/xiaoniu-578fa6bff964d005/UnbiasedWatermark
Original work: https://arxiv.org/pdf/2310.10669 by Zhengmian HU et al.
"""

class WatermarkLogitsProcessor:
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

    def _get_codes(self, context, current_pos):
        batch_size = len(context)

        if current_pos == 0:
            context_codes = [
                context[i][-self.cc_length:].tobytes() for i in range(batch_size)
            ]

        else:
            cc_pos = current_pos - self.cc_length

            if cc_pos < 0:
                cc_pos = 0
            else:
                cc_pos = cc_pos

            context_codes = [
                context[i][cc_pos:current_pos][~np.isnan(context[i][cc_pos:current_pos])].tobytes() for i in
                range(batch_size)
            ]

        mask, seeds = zip(
            *[
                (context_code in self.cc_history, self.get_rng_seed(context_code))
                for context_code in context_codes
            ]
        )
        return mask, seeds

    def __call__(self,
                 mode: str = 'normal',
                 context: np.ndarray = None,
                 logits: np.ndarray = None,
                 current_pos: int = None):

        if mode == 'normal':
            current_pos = 0
        elif mode == 'order_agnoistic':
            current_pos = current_pos
        else:
            raise NotImplementedError('Current watermark processor does not support this mode')

        mask, seeds = self._get_codes(context, current_pos=current_pos)

        rng = [
            np.random.default_rng(seed) for seed in seeds
        ]

        mask = np.array(mask)

        watermark_code = self.reweight.watermark_code_type.from_random(
            rng, logits.shape[1]
        )

        reweighted_logits = self.reweight.reweight(watermark_code, logits)

        if self.ignore_history:
            return reweighted_logits
        else:
            return np.where(mask[:, None], logits, reweighted_logits)

class WatermarkDetector:
    def __init__(
            self,
            private_key: any,
            context_code_length: int,
            reweight: AbstractReweight,
            vocab_size: int = 20,
    ):
        self.private_key = private_key
        self.cc_length = context_code_length
        self.vocab_size = vocab_size
        self.reweight = reweight

    def get_rng_seed(self, context_code: str) -> any:
        import hashlib
        m = hashlib.sha256()
        m.update(context_code)
        m.update(self.private_key)
        full_hash = m.digest()
        seed = int.from_bytes(full_hash, "big") % (2 ** 32 - 1)
        return seed

    def _get_codes(self, context):
        context_code = context[-self.cc_length:]
        return self.get_rng_seed(context_code)

    def detect(self,
               input_ids: np.ndarray):
        """
        :param input_ids: sequences after tokenization
        :return:
        """
        scores = []
        for i, _ in enumerate(input_ids):
            seed = self._get_codes(input_ids[:i])
            rng = np.random.default_rng(seed)
            watermark_code = self.reweight.watermark_code_type.from_random(rng, self.vocab_size)
            score = self.reweight.get_la_score(watermark_code)
            ...
            scores.append(score)
        return np.mean(score)
