from functools import lru_cache


def cached_property(method):
    """Quick'n'dirty implementation of cached_property coming in Python 3.8."""
    return property(lru_cache(maxsize=None)(method))
