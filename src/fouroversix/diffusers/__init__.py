"""Diffusers integration for Four Over Six NVFP4 quantization."""

import warnings

try:
    from .config import FourOverSixConfig
    from .quantizer import FourOverSixQuantizer
except ImportError:
    warnings.warn(
        "Install diffusers>=0.32.0 to use the diffusers integration: "
        "pip install 'fouroversix[diffusers]'",
        stacklevel=2,
    )

    FourOverSixConfig = None
    FourOverSixQuantizer = None


def register() -> None:
    """
    Register Four Over Six with the diffusers auto-quantizer mapping.

    After calling this function, ``FourOverSixConfig`` can be passed directly to
    ``from_pretrained(quantization_config=...)`` on any diffusers ``ModelMixin``.
    """
    if FourOverSixConfig is None or FourOverSixQuantizer is None:
        msg = (
            "Cannot register: diffusers is not installed. "
            "Install it with: pip install 'fouroversix[diffusers]'"
        )
        raise ImportError(msg)

    from diffusers.quantizers.auto import (
        AUTO_QUANTIZATION_CONFIG_MAPPING,
        AUTO_QUANTIZER_MAPPING,
    )

    AUTO_QUANTIZER_MAPPING["fouroversix"] = FourOverSixQuantizer
    AUTO_QUANTIZATION_CONFIG_MAPPING["fouroversix"] = FourOverSixConfig


if FourOverSixConfig is not None:
    register()


__all__ = ["FourOverSixConfig", "FourOverSixQuantizer", "register"]
