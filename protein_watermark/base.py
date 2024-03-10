from typing import Union
import numpy as np

from abc import ABC, abstractmethod


class AbstractWatermarkCode(ABC):
    @classmethod
    @abstractmethod
    def from_random(cls,
                    rng: Union[np.random.Generator, list[np.random.Generator]],
                    vocab_size: int):
        pass


class AbstractReweight(ABC):
    watermark_code_type: type[AbstractWatermarkCode]

    @abstractmethod
    def reweight(self,
                 code: AbstractWatermarkCode,
                 p: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_la_score(self,
                     code: AbstractWatermarkCode) -> np.ndarray:
        pass