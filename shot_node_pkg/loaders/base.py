from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any
import torch

from ..utils.file_utils import has_extension
from .utils.file_utils import (
    EXR_EXTENSIONS
)

# EXR loader is optional (requires OpenImageIO)
try:
    from .exr_loader import ExrLoader
    EXR_AVAILABLE = True
except ImportError:
    ExrLoader = None
    EXR_AVAILABLE = False

from .image_loader import ImageLoader


def resolve_loader(file_path: str):
    """Get appropriate loader for file type."""
    if has_extension(file_path, EXR_EXTENSIONS):
        if not EXR_AVAILABLE:
            raise ImportError(
                "OpenImageIO is required for EXR files but not available. "
                "Install with: pip install OpenImageIO"
            )
        return ExrLoader()
    else:
        return ImageLoader()



@dataclass
class LoadResult:
    image: torch.Tensor  # [1, H, W, 3] RGB tensor
    mask: torch.Tensor   # [1, H, W] mask/alpha tensor
    layers: Dict[str, torch.Tensor] = field(default_factory=dict)  # Additional layers (EXR)
    cryptomatte: Dict[str, torch.Tensor] = field(default_factory=dict)  # Cryptomatte layers (EXR)
    metadata: Dict[str, Any] = field(default_factory=dict)  # File metadata
    width: int = 0
    height: int = 0
    has_alpha: bool = False


class BaseLoader(ABC):

    EXTENSIONS: set = set()

    @abstractmethod
    def load_image(self, file_path: str, normalize: bool = False) -> LoadResult:
        """
        Load a single image file.

        Args:
            file_path: Path to the image file
            normalize: If True, normalize values to 0-1 range (useful for HDR)

        Returns:
            LoadResult with image data and metadata
        """
        pass

    @abstractmethod
    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata from file without loading full image data.

        Args:
            file_path: Path to the image file

        Returns:
            Dictionary with file metadata (dimensions, channels, etc.)
        """
        pass

    @classmethod
    def can_load(cls, file_path: str) -> bool:
        return has_extension(file_path, cls.EXTENSIONS)

    @staticmethod
    def resize_to_match(image: torch.Tensor, target_width: int, target_height: int) -> torch.Tensor:
        """
        Resize image tensor to match target dimensions.

        Args:
            image: Input tensor [1, H, W, C] or [H, W, C]
            target_width: Target width
            target_height: Target height

        Returns:
            Resized tensor
        """
        from comfy.utils import common_upscale

        needs_squeeze = False
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            needs_squeeze = True

        # common_upscale expects [B, C, H, W]
        t = image.movedim(-1, 1)
        t = common_upscale(t, target_width, target_height, "lanczos", "center")
        result = t.movedim(1, -1)

        if needs_squeeze:
            result = result.squeeze(0)

        return result
