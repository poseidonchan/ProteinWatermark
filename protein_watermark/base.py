from typing import Union, List, Type
import numpy as np

from abc import ABC, abstractmethod

"""
Codes below are modified from unbiased watermark, 
GitHub repo: https://github.com/xiaoniu-578fa6bff964d005/UnbiasedWatermark
Original work: https://arxiv.org/pdf/2310.10669 by Zhengmian HU et al.
"""

class AbstractWatermarkCode(ABC):
    @classmethod
    @abstractmethod
    def from_random(cls,
                    rng: Union[np.random.Generator, List[np.random.Generator]],
                    vocab_size: int):
        pass


class AbstractReweight(ABC):
    watermark_code_type: Type[AbstractWatermarkCode]

    @abstractmethod
    def reweight(self,
                 code: AbstractWatermarkCode,
                 p: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_la_score(self,
                     code: AbstractWatermarkCode) -> np.ndarray:
        pass