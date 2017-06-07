# -*- coding: utf-8 -*-

__version__ = "0.0.1.dev0"

try:
    __FINUFFT_SETUP__
except NameError:
    __FINUFFT_SETUP__ = False

if not __FINUFFT_SETUP__:
    __all__ = ["nufft1d1", "nufft1d2", "nufft1d3"]

    from .interface import (
        nufft1d1, nufft1d2, nufft1d3,
    )
