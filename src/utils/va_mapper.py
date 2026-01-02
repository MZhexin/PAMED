# src/utils/va_mapper.py
import numpy as np


class VAMapper:
    """
    VA Linear Mapping Converter

    Function: Handle linear mapping between IADS-E VA values and MemoMusic VA values

    Attributes:
        iads_range (tuple): VA range for IADS-E
        memo_v_range (tuple): Valence range for MemoMusic
        memo_a_range (tuple): Arousal range for MemoMusic
    """

    def __init__(self):
        self.iads_range = (1, 9)
        self.memo_v_range = (-5, 5)
        self.memo_a_range = (0, 10)

    def map_to_memo(self, v_iads: float, a_iads: float) -> tuple[float, float]:
        """
        Map IADS-E VA values to MemoMusic range
        VA_memo = (VA_iads - 1) / (VA_iads_max - VA_iads_min) × (VA_memo_max - VA_memo_min) + VA_memo_min

        Params:
            v_iads: IADS-E Valence value
            a_iads: IADS-E Arousal value

        Returns:
            (v_memo, a_memo): Mapped MemoMusic VA values
        """
        # Valence
        v_memo = ((v_iads - self.iads_range[0]) / (self.iads_range[1] - self.iads_range[0])) * \
                 (self.memo_v_range[1] - self.memo_v_range[0]) + self.memo_v_range[0]

        # Arousal
        a_memo = ((a_iads - self.iads_range[0]) / (self.iads_range[1] - self.iads_range[0])) * \
                 (self.memo_a_range[1] - self.memo_a_range[0]) + self.memo_a_range[0]

        return round(v_memo, 2), round(a_memo, 2)