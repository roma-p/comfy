"""
OpenImageIO-based EXR loader for multi-layer EXR files
"""

import os
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple

from .base import BaseLoader, LoadResult
from ..utils.file_utils import EXR_EXTENSIONS

# Check for OpenImageIO availability
try:
    import OpenImageIO as oiio
    OIIO_AVAILABLE = True
except ImportError:
    OIIO_AVAILABLE = False
    oiio = None


class ExrLoader(BaseLoader):
    """Loader for EXR files using OpenImageIO."""

    EXTENSIONS = EXR_EXTENSIONS

    def __init__(self):
        if not OIIO_AVAILABLE:
            raise ImportError(
                "OpenImageIO is required for EXR loading but not available. "
                "Install it with: pip install OpenImageIO"
            )

    def load_image(self, file_path: str, normalize: bool = False) -> LoadResult:
        """
        Load a single EXR file with all layers.

        Args:
            file_path: Path to the EXR file
            normalize: If True, normalize HDR values to 0-1 range

        Returns:
            LoadResult with image data, layers, and cryptomatte
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"EXR file not found: {file_path}")

        # Scan metadata first
        metadata = self._scan_metadata(file_path)

        # Load all pixel data
        all_subimage_data = self._load_all_data(file_path)

        # Process into tensors
        layers_dict = {}
        cryptomatte_dict = {}
        rgb_tensor = None
        alpha_tensor = None

        for subimage_idx, subimage_info in enumerate(metadata["subimages"]):
            if subimage_idx not in all_subimage_data:
                continue

            subimage_data = all_subimage_data[subimage_idx]
            channel_names = subimage_info["channel_names"]
            height, width = subimage_info["height"], subimage_info["width"]

            # Process main subimage (index 0) for RGB/Alpha
            if subimage_idx == 0:
                rgb_tensor, alpha_tensor = self._process_default_channels(
                    subimage_data, channel_names, height, width, normalize
                )

                # Process additional layers in first subimage
                channel_groups = self._get_channel_groups(channel_names)
                self._process_layer_groups(
                    channel_groups, subimage_data, channel_names,
                    normalize, layers_dict, cryptomatte_dict
                )

            # Process named subimages as additional layers
            subimage_name = subimage_info.get("name", "default")
            if subimage_name != "default":
                self._process_named_subimage(
                    subimage_name, subimage_data, normalize, layers_dict
                )

        # Ensure we have valid tensors
        if rgb_tensor is None:
            raise ValueError(f"Could not extract RGB data from EXR: {file_path}")
        if alpha_tensor is None:
            alpha_tensor = torch.ones((1, metadata["subimages"][0]["height"],
                                       metadata["subimages"][0]["width"]))

        return LoadResult(
            image=rgb_tensor,
            mask=alpha_tensor,
            layers=layers_dict,
            cryptomatte=cryptomatte_dict,
            metadata=metadata,
            width=metadata["subimages"][0]["width"],
            height=metadata["subimages"][0]["height"],
            has_alpha="A" in metadata["subimages"][0]["channel_names"],
        )

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get metadata from EXR file without loading pixel data."""
        return self._scan_metadata(file_path)

    def _scan_metadata(self, file_path: str) -> Dict[str, Any]:
        """Scan EXR file for metadata without loading pixels."""
        input_file = oiio.ImageInput.open(file_path)
        if not input_file:
            raise IOError(f"Could not open EXR file: {file_path}")

        try:
            metadata = {"subimages": [], "file_path": file_path}
            current_subimage = 0
            more_subimages = True

            while more_subimages:
                spec = input_file.spec()

                subimage_info = {
                    "index": current_subimage,
                    "name": spec.getattribute("name") or "default",
                    "width": spec.width,
                    "height": spec.height,
                    "channels": spec.nchannels,
                    "channel_names": [spec.channel_name(i) for i in range(spec.nchannels)],
                }
                metadata["subimages"].append(subimage_info)

                more_subimages = input_file.seek_subimage(current_subimage + 1, 0)
                current_subimage += 1

            metadata["is_multipart"] = len(metadata["subimages"]) > 1
            metadata["subimage_count"] = len(metadata["subimages"])
            return metadata

        finally:
            input_file.close()

    def _load_all_data(self, file_path: str) -> Dict[int, np.ndarray]:
        """Load all pixel data from all subimages."""
        input_file = oiio.ImageInput.open(file_path)
        if not input_file:
            raise IOError(f"Could not open EXR file: {file_path}")

        try:
            all_data = {}
            current_subimage = 0
            more_subimages = True

            while more_subimages:
                spec = input_file.spec()
                pixels = input_file.read_image()

                if pixels is not None:
                    all_data[current_subimage] = np.array(
                        pixels, dtype=np.float32
                    ).reshape(spec.height, spec.width, spec.nchannels)

                more_subimages = input_file.seek_subimage(current_subimage + 1, 0)
                current_subimage += 1

            return all_data

        finally:
            input_file.close()

    def _process_default_channels(
        self, data: np.ndarray, channel_names: List[str],
        height: int, width: int, normalize: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract RGB and Alpha from default channels."""

        # Try to find RGB channels
        if all(c in channel_names for c in ['R', 'G', 'B']):
            r_idx = channel_names.index('R')
            g_idx = channel_names.index('G')
            b_idx = channel_names.index('B')
            rgb_array = np.stack([
                data[:, :, r_idx],
                data[:, :, g_idx],
                data[:, :, b_idx]
            ], axis=2)
        elif data.shape[2] >= 3:
            rgb_array = data[:, :, :3]
        else:
            rgb_array = np.stack([data[:, :, 0]] * 3, axis=2)

        rgb_tensor = torch.from_numpy(rgb_array).float().unsqueeze(0)

        if normalize:
            rgb_min, rgb_max = rgb_tensor.min(), rgb_tensor.max()
            if rgb_max - rgb_min > 0:
                rgb_tensor = (rgb_tensor - rgb_min) / (rgb_max - rgb_min)

        # Try to find Alpha channel
        if 'A' in channel_names:
            a_idx = channel_names.index('A')
            alpha_array = data[:, :, a_idx]
            alpha_tensor = torch.from_numpy(alpha_array).float().unsqueeze(0)
            if normalize:
                alpha_tensor = alpha_tensor.clamp(0, 1)
        else:
            alpha_tensor = torch.ones((1, height, width))

        return rgb_tensor, alpha_tensor

    def _get_channel_groups(self, channel_names: List[str]) -> Dict[str, List[str]]:
        """Group channel names by their prefix (before the dot)."""
        groups = {}

        for channel in channel_names:
            if '.' in channel:
                parts = channel.split('.')
                if len(parts) > 2:
                    prefix = '.'.join(parts[:-1])
                    suffix = parts[-1]
                else:
                    prefix, suffix = parts[0], parts[1]

                if prefix not in groups:
                    groups[prefix] = []
                groups[prefix].append(suffix)
            else:
                if channel not in groups:
                    groups[channel] = []
                groups[channel].append(None)

        return groups

    def _is_cryptomatte_layer(self, name: str) -> bool:
        """Check if layer name indicates a cryptomatte layer."""
        name_lower = name.lower()
        return (
            "cryptomatte" in name_lower or
            name_lower.startswith("crypto") or
            any(k in name_lower for k in ["cryptoasset", "cryptomaterial", "cryptoobject"])
        )

    def _process_layer_groups(
        self, channel_groups: Dict[str, List[str]],
        data: np.ndarray, channel_names: List[str],
        normalize: bool, layers_dict: Dict, cryptomatte_dict: Dict
    ):
        """Process layer groups into tensors."""

        for group_name, suffixes in channel_groups.items():
            # Skip default channels
            if group_name in ('R', 'G', 'B', 'A'):
                continue

            is_crypto = self._is_cryptomatte_layer(group_name)

            # Find channel indices for this group
            group_indices = []
            for i, ch in enumerate(channel_names):
                if ch == group_name or ch.startswith(f"{group_name}."):
                    group_indices.append(i)

            if not group_indices:
                continue

            # Process based on channel pattern
            if all(s in suffixes for s in ['R', 'G', 'B']):
                self._process_rgb_layer(
                    group_name, 'R', 'G', 'B', channel_names, data,
                    normalize, is_crypto, layers_dict, cryptomatte_dict
                )
            elif all(s in suffixes for s in ['r', 'g', 'b']):
                self._process_rgb_layer(
                    group_name, 'r', 'g', 'b', channel_names, data,
                    normalize, is_crypto, layers_dict, cryptomatte_dict
                )
            elif all(s in suffixes for s in ['X', 'Y', 'Z']):
                self._process_xyz_layer(
                    group_name, 'X', 'Y', 'Z', channel_names, data,
                    normalize, layers_dict
                )
            elif len(group_indices) == 1:
                self._process_single_channel(
                    group_name, group_indices[0], data, normalize, layers_dict
                )
            else:
                self._process_multi_channel(
                    group_name, group_indices, data, normalize,
                    is_crypto, layers_dict, cryptomatte_dict
                )

    def _process_rgb_layer(
        self, group_name: str, r: str, g: str, b: str,
        channel_names: List[str], data: np.ndarray,
        normalize: bool, is_crypto: bool,
        layers_dict: Dict, cryptomatte_dict: Dict
    ):
        """Process an RGB layer."""
        try:
            r_idx = channel_names.index(f"{group_name}.{r}")
            g_idx = channel_names.index(f"{group_name}.{g}")
            b_idx = channel_names.index(f"{group_name}.{b}")
        except ValueError:
            return

        rgb_array = np.stack([
            data[:, :, r_idx],
            data[:, :, g_idx],
            data[:, :, b_idx]
        ], axis=2)

        tensor = torch.from_numpy(rgb_array).float().unsqueeze(0)

        if normalize:
            t_min, t_max = tensor.min(), tensor.max()
            if t_max - t_min > 0:
                tensor = (tensor - t_min) / (t_max - t_min)

        if is_crypto:
            cryptomatte_dict[group_name] = tensor
        else:
            layers_dict[group_name] = tensor

    def _process_xyz_layer(
        self, group_name: str, x: str, y: str, z: str,
        channel_names: List[str], data: np.ndarray,
        normalize: bool, layers_dict: Dict
    ):
        """Process an XYZ vector layer."""
        try:
            x_idx = channel_names.index(f"{group_name}.{x}")
            y_idx = channel_names.index(f"{group_name}.{y}")
            z_idx = channel_names.index(f"{group_name}.{z}")
        except ValueError:
            return

        xyz_array = np.stack([
            data[:, :, x_idx],
            data[:, :, y_idx],
            data[:, :, z_idx]
        ], axis=2)

        tensor = torch.from_numpy(xyz_array).float().unsqueeze(0)

        if normalize:
            max_abs = tensor.abs().max()
            if max_abs > 0:
                tensor = tensor / max_abs

        layers_dict[group_name] = tensor

    def _process_single_channel(
        self, group_name: str, channel_idx: int,
        data: np.ndarray, normalize: bool, layers_dict: Dict
    ):
        """Process a single channel as a mask."""
        channel_array = data[:, :, channel_idx]
        tensor = torch.from_numpy(channel_array).float().unsqueeze(0)

        if normalize:
            t_min, t_max = tensor.min(), tensor.max()
            if t_max - t_min > 0:
                tensor = (tensor - t_min) / (t_max - t_min)

        layers_dict[group_name] = tensor

    def _process_multi_channel(
        self, group_name: str, indices: List[int],
        data: np.ndarray, normalize: bool, is_crypto: bool,
        layers_dict: Dict, cryptomatte_dict: Dict
    ):
        """Process multi-channel data."""
        channels_to_use = min(3, len(indices))
        arrays = [data[:, :, indices[i]] for i in range(channels_to_use)]

        # Pad to 3 channels if needed
        while len(arrays) < 3:
            arrays.append(arrays[-1])

        multi_array = np.stack(arrays, axis=2)
        tensor = torch.from_numpy(multi_array).float().unsqueeze(0)

        if normalize:
            t_min, t_max = tensor.min(), tensor.max()
            if t_max - t_min > 0:
                tensor = (tensor - t_min) / (t_max - t_min)

        if is_crypto:
            cryptomatte_dict[group_name] = tensor
        else:
            layers_dict[group_name] = tensor

    def _process_named_subimage(
        self, name: str, data: np.ndarray,
        normalize: bool, layers_dict: Dict
    ):
        """Process a named subimage as a layer."""
        channels = data.shape[2]

        if channels >= 3:
            rgb_array = data[:, :, :3]
            tensor = torch.from_numpy(rgb_array).float().unsqueeze(0)
        elif channels == 1:
            tensor = torch.from_numpy(data[:, :, 0]).float().unsqueeze(0)
        else:
            return

        if normalize:
            t_min, t_max = tensor.min(), tensor.max()
            if t_max - t_min > 0:
                tensor = (tensor - t_min) / (t_max - t_min)

        layers_dict[name] = tensor
