"""
utilities for code development of back test platform and
trading modules
"""

from enum import Enum

class Side(Enum):
    """Used to express signals and trade direction in natural language"""
    LONG = 1
    SHORT = -1
    CLOSE = 0

    @classmethod
    def _missing_(cls, value):
        """
        provides additional mappings of enum values
        """
        enum_map = {
            'BUY': cls.LONG,
            'SELL_SHORT': cls.SHORT
        }
        if (res := enum_map.get(value, None)) is not None:
            res = super()._missing_(value)
        return res


