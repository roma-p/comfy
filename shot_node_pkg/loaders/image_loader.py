import os
import numpy as np
import torch
from PIL import Image, ImageOps
from typing import Dict, Any

from .base import BaseLoader, LoadResult
from ..utils.file_utils import IMG_EXTENSIONS


class ImageLoader(BaseLoader):

    EXTENSIONS = IMG_EXTENSIONS

    def load_image(self, file_path: str, normalize: bool = False) -> LoadResult:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        img = Image.open(file_path)
        img = ImageOps.exif_transpose(img)

        width, height = img.size
        original_mode = img.mode
        has_alpha = 'A' in img.getbands()

        # Determine bit depth from mode
        bit_depth = self._get_bit_depth(original_mode)

        # Get color profile if present
        color_profile = None
        if 'icc_profile' in img.info:
            color_profile = "embedded"  # Has ICC profile
        elif original_mode in ('RGB', 'RGBA'):
            color_profile = "sRGB"  # Assumed for standard RGB images

        if has_alpha:
            img = img.convert("RGBA")
        else:
            img = img.convert("RGB")

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
                "mode": original_mode,
                "bit_depth": bit_depth,
                "color_profile": color_profile,
            },
            width=width,
            height=height,
            has_alpha=has_alpha,
        )

    def _get_bit_depth(self, mode: str) -> int:
        """Get bit depth per channel from PIL mode."""
        bit_depths = {
            '1': 1,      # 1-bit pixels, black and white
            'L': 8,      # 8-bit grayscale
            'P': 8,      # 8-bit palette
            'RGB': 8,    # 8-bit RGB
            'RGBA': 8,   # 8-bit RGBA
            'CMYK': 8,   # 8-bit CMYK
            'YCbCr': 8,  # 8-bit YCbCr
            'LAB': 8,    # 8-bit LAB
            'HSV': 8,    # 8-bit HSV
            'I': 32,     # 32-bit signed integer
            'F': 32,     # 32-bit float
            'I;16': 16,  # 16-bit unsigned integer
            'I;16L': 16,
            'I;16B': 16,
        }
        return bit_depths.get(mode, 8)

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        img = Image.open(file_path)
        img = ImageOps.exif_transpose(img)

        color_profile = None
        if 'icc_profile' in img.info:
            color_profile = "embedded"
        elif img.mode in ('RGB', 'RGBA'):
            color_profile = "sRGB"

        return {
            "file_path": file_path,
            "width": img.size[0],
            "height": img.size[1],
            "format": img.format or "unknown",
            "mode": img.mode,
            "bit_depth": self._get_bit_depth(img.mode),
            "color_profile": color_profile,
            "has_alpha": 'A' in img.getbands(),
        }
