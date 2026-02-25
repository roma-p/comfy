"""
Loaders module - Image loading backends for ReadNode
"""

from .base import BaseLoader, LoadResult, EXR_AVAILABLE, resolve_loader
from .image_loader import ImageLoader

__all__ = [
    'BaseLoader',
    'LoadResult',
    'ImageLoader',
    'ExrLoader',
    'EXR_AVAILABLE',
    'resolve_loader',
]
