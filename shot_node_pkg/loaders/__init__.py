"""
Loaders module - Image loading backends for ReadNode
"""

from .base import BaseLoader, LoadResult
from .image_loader import ImageLoader

# EXR loader is optional (requires OpenImageIO)
try:
    from .exr_loader import ExrLoader
    EXR_AVAILABLE = True
except ImportError:
    ExrLoader = None
    EXR_AVAILABLE = False

__all__ = ['BaseLoader', 'LoadResult', 'ImageLoader', 'ExrLoader', 'EXR_AVAILABLE']
