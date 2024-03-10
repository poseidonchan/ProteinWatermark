import numpy as np
from typing import Union
from .base import AbstractReweight, AbstractWatermarkCode

def get_gumbel_variables(rng: np.random.Generator,
                         vocab_size: int):
    u = rng.random((vocab_size,))  # ~ Unifom(0, 1)
    e = -np.log(u)  # ~ Exp(1)
    g = -np.log(e)  # ~ Gumbel(0, 1)
    return u, e, g


class DeltaGumbel_WatermarkCode(AbstractWatermarkCode):
    def __init__(self, g: np.ndarray):
        self.g = g

    @classmethod
    def from_random(
            cls,
            rng: Union[np.random.Generator, list[np.random.Generator]],
            vocab_size: int,
    ):
        if isinstance(rng, list):
            batch_size = len(rng)
            g = np.stack(
                [get_gumbel_variables(rng[i], vocab_size)[2] for i in range(batch_size)]
            )
        else:
            g = get_gumbel_variables(rng, vocab_size)[2]

        return cls(g)


class DeltaGumbel_Reweight(AbstractReweight):
    watermark_code_type = DeltaGumbel_WatermarkCode

    def __repr__(self):
        return f"DeltaGumbel_Reweight()"

    def reweight(
            self, code: AbstractWatermarkCode, p_logits: np.ndarray
    ) -> np.ndarray:
        assert isinstance(code, DeltaGumbel_WatermarkCode)

        index = np.argmax(p_logits + code.g, axis=-1)

        mask = np.arange(p_logits.shape[-1]) == index[..., None]

        modified_logits = np.where(
            mask,
            np.full_like(p_logits, 0),
            np.full_like(p_logits, float("-inf")),
        )
        return modified_logits

    def get_la_score(self, code):
        """likelihood agnostic score"""
        return np.array(np.log(2)) - np.exp(-code.g)