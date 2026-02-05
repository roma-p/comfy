"""
PIL-based image loader for standard image formats (PNG, JPG, etc.)
"""

import os
import numpy as np
import torch
from PIL import Image, ImageOps
from typing import Dict, Any

from .base import BaseLoader, LoadResult
from ..utils.file_utils import IMG_EXTENSIONS


class ImageLoader(BaseLoader):
    """Loader for standard image formats using PIL."""

    EXTENSIONS = IMG_EXTENSIONS

    def load_image(self, file_path: str, normalize: bool = False) -> LoadResult:
        """
        Load a single image file using PIL.

        Args:
            file_path: Path to the image file
            normalize: If True, normalize values to 0-1 range (no-op for standard images)

        Returns:
            LoadResult with image data
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        # Open and process image
        img = Image.open(file_path)
        img = ImageOps.exif_transpose(img)

        width, height = img.size
        has_alpha = 'A' in img.getbands()

        # Convert to appropriate format
        if has_alpha:
            img = img.convert("RGBA")
        else:
            img = img.convert("RGB")

        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        img_array /= 255.0

        # Split RGB and Alpha
        if has_alpha:
            rgb_array = img_array[:, :, :3]
            alpha_array = 1.0 - img_array[:, :, 3]  # Invert alpha for ComfyUI mask convention
        else:
            rgb_array = img_array
            alpha_array = np.zeros((height, width), dtype=np.float32)

        # Convert to tensors with batch dimension
        image_tensor = torch.from_numpy(rgb_array).unsqueeze(0)  # [1, H, W, 3]
        mask_tensor = torch.from_numpy(alpha_array).unsqueeze(0)  # [1, H, W]

        return LoadResult(
            image=image_tensor,
            mask=mask_tensor,
            layers={},  # Standard images don't have layers
            cryptomatte={},  # Standard images don't have cryptomatte
            metadata={
                "file_path": file_path,
                "format": img.format or "unknown",
                "mode": img.mode,
            },
            width=width,
            height=height,
            has_alpha=has_alpha,
        )

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata from image file without loading full data.

        Args:
            file_path: Path to the image file

        Returns:
            Dictionary with file metadata
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        img = Image.open(file_path)
        img = ImageOps.exif_transpose(img)

        return {
            "file_path": file_path,
            "width": img.size[0],
            "height": img.size[1],
            "format": img.format or "unknown",
            "mode": img.mode,
            "has_alpha": 'A' in img.getbands(),
        }


def scan_image_sizes(file_paths: list) -> tuple:
    """
    Scan multiple images to determine common size and alpha presence.

    Args:
        file_paths: List of image file paths

    Returns:
        Tuple of (most_common_width, most_common_height, any_has_alpha)
    """
    sizes = {}
    has_alpha = False

    for image_path in file_paths:
        try:
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
            has_alpha |= 'A' in img.getbands()
            size = img.size
            sizes[size] = sizes.get(size, 0) + 1
        except Exception:
            continue

    if not sizes:
        return (512, 512, False)  # Fallback

    # Find most common size
    most_common = max(sizes.items(), key=lambda x: x[1])[0]
    return (most_common[0], most_common[1], has_alpha)
